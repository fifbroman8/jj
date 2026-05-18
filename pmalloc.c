/* A general purpose memory allocator which uses THP */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <signal.h>
#include <search.h>
#include <assert.h>
#include <time.h>
#include <pthread.h>
#include <dlfcn.h>
#include <malloc.h>
#include <stdatomic.h>
#include <sys/sysinfo.h>

#define HUGEPAGE (1024*2048)
#define MAX_MEM (32UL*1024*1024*1024)
#define MIN_BIT_OFFSET 4 //min size is 16B, should <= 6
#define ASSERT(x) //assert(x)
#define MAX_FREED_BLOCK 32

#define likely(x) __builtin_expect (!!(x), 1)
#define unlikely(x) __builtin_expect (!!(x), 0)
#define PREFETCH(x) __builtin_prefetch(x)
#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#endif
extern void *__libc_malloc(size_t);
extern void *__libc_free(void *);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_reallocarray(void *, size_t, size_t);

#define libc_malloc(len)  __libc_malloc(len)
#define libc_free(ptr)  __libc_free(ptr)
#define libc_realloc(ptr, len) __libc_realloc(ptr, len)
#define libc_calloc(n, m)  __libc_calloc(n,m)
#define libc_reallocarray(ptr, n, m)  __libc_reallocarray(ptr, n, m)

typedef struct slist{
    struct slist *next;
}slist_t;

typedef struct mem_header{
    void *block;
    struct mem_header *next;
    slist_t *hdr;  //header of freed memory slice
    uint64_t freed_slice;
    uint64_t total_slice_pow; //total_slice = 1U<<total_slice_pow
    uint64_t slice_size_pow; //slice_size = 1U<<slice_size_pow
    uint64_t total_slice;
    volatile uint64_t __lock;
} mem_header_t;

_Static_assert(sizeof(mem_header_t) == 64);

typedef struct{
    void *start;
    uint64_t len;
    uint64_t block_size; //block size
    long freed_block_count;
    long mapped_block_count;
    uint64_t block_pow; //precalculate value of ffsl(allocator.block)-1
    uint64_t pad1[2];
    volatile uint64_t __lock;
    uint64_t pad2[7];
    volatile uint64_t __slow_path_lock;
    uint64_t pad3[7];
    uint64_t block_counter[32];
    mem_header_t mem_header[MAX_MEM/HUGEPAGE] __attribute__((aligned(64)));
    mem_header_t *mem_hdr_in_used;
} mem_block_allocator;

static mem_block_allocator allocator;

static __thread mem_header_t *freed_header_list[32]; //cache for mem_header_t, first pointer points to 16B header, second points to 32B

#define PMALLOC_BLOCK() while (__sync_lock_test_and_set(&allocator.__lock, 1))
#define PMALLOC_BUNLOCK() __sync_lock_release(&allocator.__lock)
#define PMALLOC_SLOCK() while (__sync_lock_test_and_set(&allocator.__slow_path_lock, 1))
#define PMALLOC_SUNLOCK() __sync_lock_release(&allocator.__slow_path_lock)
#define PMALLOC_MLOCK(m) while (__sync_lock_test_and_set(&m->__lock, 1))
#define PMALLOC_MUNLOCK(m) __sync_lock_release(&m->__lock)

static inline uint64_t next_pow2(uint64_t x) {
    //roundup to at least 16
    return x <= (1U<<MIN_BIT_OFFSET) ? MIN_BIT_OFFSET : (64-__builtin_clzl(x-1));
}

static inline void* alloc_one_block(uint64_t slice_size_pow)
{
    PMALLOC_BLOCK();
    int mapped_block_count = allocator.mapped_block_count++;
    uint64_t offset = mapped_block_count * allocator.block_size;
    if(unlikely(offset > allocator.len)){
        --allocator.mapped_block_count;
        PMALLOC_BUNLOCK();
        return NULL;
    }
    allocator.block_counter[slice_size_pow]++;
    PMALLOC_BUNLOCK();
    return allocator.start + offset;
}

static inline void free_unused_block(mem_header_t *header)
{
    if(unlikely(header->freed_slice == header->total_slice)){
        header->hdr = NULL;
        madvise(header->block, HUGEPAGE, MADV_DONTNEED);
    }
}

static inline void *alloc_one_slice(mem_header_t *header)
{
    //ASSERT(header->freed_slice);
    PMALLOC_MLOCK(header);
    slist_t *ptr = header->hdr;
    if(ptr){
        header->hdr = ptr->next;
        header->freed_slice--;
    }else if(header->freed_slice){
        ptr = (slist_t*)((uint64_t)header->block + ((header->total_slice - header->freed_slice)<<header->slice_size_pow));
        header->freed_slice--;
    }

    PMALLOC_MUNLOCK(header);
    return (void*)ptr;
}

static inline void free_one_slice(mem_header_t *header, void *ptr)
{
    ASSERT(header->freed_slice < header->total_slice);
    ASSERT((uint64_t)ptr - (uint64_t)header->block < allocator.block_size);
    PMALLOC_MLOCK(header);
    ((slist_t*)ptr)->next = header->hdr;
    header->hdr = ptr;
    header->freed_slice++;
    free_unused_block(header);
    PMALLOC_MUNLOCK(header);
}

static void *__pmalloc(size_t len)
{
    uint64_t pow = next_pow2(len);
    //fast path
    mem_header_t *header = freed_header_list[pow - MIN_BIT_OFFSET];
    if(header && header->slice_size_pow == pow){
        if(likely(header->freed_slice)){
            void *ptr = alloc_one_slice(header);
            if(ptr){
                return ptr;
            }
        }
    }

    //slow path
    header = allocator.mem_hdr_in_used;
    while(header){
        PREFETCH(header->next);
        if(header->slice_size_pow == pow && header->freed_slice){
            void *ptr = alloc_one_slice(header);
            if(likely(header->freed_slice)){
                freed_header_list[pow - MIN_BIT_OFFSET] = header;
            }
            if(ptr){
                return ptr;
            }
        }
        if(header->next){
            header = header->next;
            continue;
        }else{
            PMALLOC_SLOCK();
            if(header->next){
                header = header->next;
                PMALLOC_SUNLOCK();
                continue;
            }
            mem_header_t *new_header = NULL;
            void *b = alloc_one_block(pow);
            if(b){
                int idx = (b - allocator.start)>>allocator.block_pow;
                new_header = &allocator.mem_header[idx];
                new_header->total_slice_pow = allocator.block_pow - pow;
                new_header->slice_size_pow = pow;
                new_header->freed_slice = new_header->total_slice = (1U<<new_header->total_slice_pow);
                new_header->block = b;
                header->next = new_header;
            }else{
                PMALLOC_SUNLOCK();
                return NULL;
            }
            freed_header_list[pow - MIN_BIT_OFFSET] = new_header;
            PMALLOC_SUNLOCK();
            void *ptr = alloc_one_slice(new_header);
            return ptr;
        }
    }
    return NULL;
}

void *pmalloc(size_t len)
{
    if(unlikely(!allocator.start)) return libc_malloc(len);
    if(unlikely(len > allocator.block_size)) return libc_malloc(len); //switch to glibc malloc for large memory request
    void *ptr = __pmalloc(len);
    if(unlikely(!ptr)) ptr = libc_malloc(len);
    return ptr;
}

static inline void __pfree(void *ptr)
{
    uint64_t index = ((uint64_t)ptr - (uint64_t)allocator.start)>>(allocator.block_pow);
    mem_header_t *header = &allocator.mem_header[index];
    ASSERT((uint64_t)ptr - (uint64_t)header->block < allocator.block_size);
    free_one_slice(header, ptr);
}

void pfree(void *ptr)
{
    if(unlikely(!allocator.start)) {
        libc_free(ptr);
        return;
    }
    if(unlikely((uint64_t)ptr - (uint64_t)allocator.start >= allocator.len )){
        libc_free(ptr);
        return;
    }
    __pfree(ptr);
}

void *pcalloc(size_t nmemb, size_t size)
{
    if(unlikely(!allocator.start)) return libc_calloc(nmemb, size);
    if(unlikely((!nmemb || !size))) return NULL;
    size_t len = nmemb*size;
    if(unlikely(len > UINT32_MAX)) return NULL;
    void *ptr = pmalloc(len);
    if(ptr) memset(ptr, 0, len);
    return ptr;
}

void *prealloc(void *ptr, size_t size)
{
    if(unlikely(!allocator.start)) return libc_realloc(ptr, size);
    if(unlikely((uint64_t)ptr - (uint64_t)allocator.start >= allocator.len )) return libc_realloc(ptr, size);

    if(size == 0){
        pfree(ptr);
        return NULL;
    }
    uint64_t index = ((uint64_t)ptr - (uint64_t)allocator.start) >> allocator.block_pow;
    mem_header_t *header = &allocator.mem_header[index];
    if(size<= (allocator.block_size>>header->total_slice_pow)){
        return ptr;
    }
    void *new_ptr = pmalloc(size);
    if(new_ptr && ptr) {
        size_t min = (uint64_t)allocator.start+allocator.len-(uint64_t)ptr;
        min = MIN(min, size);
        memcpy(new_ptr, ptr, min);
        pfree(ptr);
    }
    return new_ptr;
}

void *preallocarray(void *ptr, size_t nmemb, size_t size)
{
    if(unlikely(!allocator.start)) return libc_reallocarray(ptr, nmemb, size);
    if(unlikely(nmemb*size > UINT32_MAX)) return NULL;
    return prealloc(ptr, nmemb*size);
}

size_t pmalloc_usable_size (void *ptr)
{
    static size_t   (*__malloc_usable_size)(void*);
    if(!__malloc_usable_size){
        __malloc_usable_size = dlsym(RTLD_DEFAULT, "__malloc_usable_size");
    }
    if(unlikely(!allocator.start)) {
        return __malloc_usable_size?__malloc_usable_size(ptr):0;
    }
    if(unlikely((uint64_t)ptr - (uint64_t)allocator.start >= allocator.len )){
        return __malloc_usable_size?__malloc_usable_size(ptr):0;
    }
    uint64_t index = ((uint64_t)ptr - (uint64_t)allocator.start)>>(allocator.block_pow);
    mem_header_t *header = &allocator.mem_header[index];
    ASSERT((uint64_t)ptr - (uint64_t)header->block < allocator.block_size);
    return 1U << header->slice_size_pow;
}

void *malloc(size_t) __attribute__ ((alias ("pmalloc")));
void free(void*) __attribute__ ((alias ("pfree")));
void *realloc(void*, size_t) __attribute__ ((alias ("prealloc")));
void *calloc(size_t, size_t) __attribute__ ((alias ("pcalloc")));
void *reallocarray(void*, size_t, size_t) __attribute__ ((alias ("preallocarray")));
size_t malloc_usable_size (void *ptr) __attribute__ ((alias ("pmalloc_usable_size")));

static void pmalloc_allocator_init(void *start, uint64_t len)
{
    //memset(&allocator, 0, sizeof(allocator));
    allocator.start = start;
    allocator.len = len;
    allocator.block_size = HUGEPAGE;
    allocator.block_pow = ffsl(allocator.block_size)-1;
    uint64_t total_pointer = (ffsl(allocator.block_size) - MIN_BIT_OFFSET); //freed_header_list
    //freed_header_list = (mem_header_t**)(__libc_calloc(total_pointer, sizeof(void*)));

    //pre-allocate one block with size 16
    allocator.mapped_block_count++;
    allocator.mem_hdr_in_used = &allocator.mem_header[0];
    allocator.block_counter[4]++;
    mem_header_t *new_header = &allocator.mem_header[0];
    new_header->total_slice_pow = allocator.block_pow - 4;
    new_header->slice_size_pow = 4;
    new_header->block = allocator.start;
    new_header->freed_slice = new_header->total_slice =(1U<<new_header->total_slice_pow);
    freed_header_list[new_header->slice_size_pow - MIN_BIT_OFFSET] = new_header;
}

__attribute__((constructor)) static void __pmalloc_init()
{
    struct sysinfo info;
    sysinfo(&info);
    unsigned long len = info.totalram & ~(1024UL*1024*1024-1);
    len = MIN(len, MAX_MEM);
    void *ptr = mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, HUGEPAGE);
    if(ptr == MAP_FAILED){
        len -= 1024UL*1024*1024;
        ptr = mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if(ptr == MAP_FAILED){
            perror("Failed to alloc memory region, fallback to glibc malloc:\n");
            return;
        }
    }

    madvise(ptr, len, MADV_HUGEPAGE);
    pmalloc_allocator_init(ptr, len);
    //perror("pmalloc activate!\n");
}
