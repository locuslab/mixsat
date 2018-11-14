#include <stdio.h>
#define __USE_XOPEN
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <xmmintrin.h>
#include <smmintrin.h>

#include <time.h>
#ifndef __unix__
#include <sys/time.h>
#endif

#ifndef aligned_alloc
#define aligned_alloc Calloc
#endif

const double MEPS = 1e-24;

#define NS_PER_SEC 1000000000
int64_t wall_clock_ns()
{
#ifdef __unix__
	struct timespec tspec;
	int r = clock_gettime(CLOCK_MONOTONIC, &tspec);
	assert(r==0);
	return tspec.tv_sec*NS_PER_SEC + tspec.tv_nsec;
#else
	struct timeval tv;
	int r = gettimeofday( &tv, NULL );
	assert(r==0);
	return tv.tv_sec*NS_PER_SEC + tv.tv_usec*1000;
#endif
}

#define Malloc(size) malloc_or_abort((size), __LINE__)
void* malloc_or_abort(size_t size, int line)
{
    void *ptr = malloc(size);
    if(ptr == NULL) {
        fprintf(stderr, "Error not enough memory at %d (size=%zu)\n", line, size);
        exit(1);
    }
    return ptr;
}

#define Calloc(count, size) calloc_or_abort(count, (size), __LINE__)
void* calloc_or_abort(size_t count, size_t size, int line)
{
    void *ptr = calloc(count, size);
    if(ptr == NULL) {
        fprintf(stderr, "Errror not enough memory at %d (count=%zu, size=%zu)\n", line, count, size);
        exit(1);
    }
    return ptr;
}

double wall_time_diff(int64_t ed, int64_t st)
{
	return (double)(ed-st)/(double)NS_PER_SEC;
}

enum {MAXCUT=0, MAXSAT};
struct Parameter {
    int solver;
	int k;
    double eps;
    int max_iter;
    int verbose;
    int n_trial;
    FILE *fin;
    char *fin_name;
};
typedef struct Parameter Parameter;

typedef uint32_t lit_t;

struct SATProblem {
    int m; // number of clauses
    int n; // number of variables
    int nnz; // number of literals

    // cls: the clause to variable sparse matrix
    // var: the variable to clause sparse matrix
    // both with rows in zero-terminating format: [1 -2 3 4 0]
    lit_t **cls, **var;
    // the length of rows in cls and var
    int *cls_len, *var_len;

    // weight for clause
    int *cls_weight;
};
typedef struct SATProblem SATProblem;

int is_int(char *s)
{
    if (!s || *s == '\0') return 0;
    if (*s!='-' && !isdigit(*s)) return 0;
    for (s++; *s; s++) {
        if(!isdigit(*s)) return 0;
    }
    return 1;
}

int is_valid_literal(char *s, int nvar)
{
    if (!is_int(s)) return 0;
    int var = abs(atoi(s));
    return var != 0 && var <= nvar;
}

// DIMAC format
// Error detection: If cnf and all clause start with 1, then the format is wcnf.
void read_sat_problem(FILE *fin, SATProblem *prob)
{
    char *delim = " \t\r\n";
    int has_weight = 0;
    int nnz=0, n=0, m=0;

    int *var_len = NULL;
    int *cls_len = NULL;

    int clause_cnt = 1;
    int lineno = 1;

    char *line = NULL;
    size_t linecap = 0;

    for(; -1 != getline(&line, &linecap, fin); lineno++) {
        if(line[0] == '\0' || strchr("c\r\n", line[0]))
            continue;
        if(line[0] == 'p') {
            char *p = strtok(line+1, delim);
            if(!strcmp(p, "cnf")) {
                has_weight = 0;
            } else if(!strcmp(p, "wcnf")) {
                has_weight = 1;
            } else {
                fprintf(stderr, "(line %d) No format specified (cnf or wcnf).\n", lineno);
                exit(1);
            }
            // some public dataset is wcnf but marked as cnf

            p = strtok(NULL, delim);
            if(p)
                n = atoi(p);
            p = strtok(NULL, delim);
            if(p)
                m = atoi(p);
            if(m == 0 || n == 0) {
                fprintf(stderr, "(line %d) Wrong format in parameter line\n", lineno);
                exit(1);
            }
            cls_len = Calloc(m+1, sizeof *cls_len);
            var_len = Calloc(n+1, sizeof *var_len);
            continue;
        }
        if(var_len == NULL || cls_len == NULL) {
            fprintf(stderr, "(line %d) Clause appears before parameters\n", lineno);
            exit(1);
        }

        // count nnz and check has_weight
        int has_error = 0;
        char *p = strtok(line, delim);
        if(!p){
            fprintf(stderr, "(line %d) Empty line in clause\n", lineno);
            exit(1);
        }
        if(!strncmp(p, "1 %", 3)) // ending symbol in some format
            break;
        if(has_weight){
            if(!is_int(p) || *p=='-' || !strcmp(p, "0")){
                fprintf(stderr, "(line %d) Only accept positve integer weight\n", lineno);
                exit(1);
            }
            p = strtok(NULL, delim);
        }
        
        int is_zero_terminated = 0;

        for(; p; ){
            if(!strcmp(p, "0")){
                is_zero_terminated = 1;
                break;
            }
            nnz++;
            if(!is_valid_literal(p, n)){
                has_error = 1;
                break;
            }else{
                int var = abs(atoi(p));
                var_len[var]++;
                cls_len[clause_cnt]++;
            }
            p = strtok(NULL, delim);
        }
        if(cls_len[clause_cnt] == 0){
            fprintf(stderr, "(line %d) Clause has no literal\n", lineno);
            exit(1);
        }
        if(!is_zero_terminated){
            fprintf(stderr, "(line %d) Clause need to be terminated with 0\n", lineno);
        }
        if(has_error) {
            fprintf(stderr, "(line %d) Wrong format in clause\n", lineno);
            exit(1);
        }
        clause_cnt++;
        if(clause_cnt > m+1){
            fprintf(stderr, "(line %d) More clause then speficied in parameters (%d)\n", lineno, m);
            exit(1);
        }
    }
    if(clause_cnt < m+1){ // include the 0-th clause
        fprintf(stderr, "(error) Fewer clauses (%d) then speficied in parameters (%d) \n", clause_cnt, m);
        exit(1);
    }

    lit_t *var_pool = (lit_t*) Calloc(nnz+n + m+1, sizeof *var_pool);
    lit_t *cls_pool = (lit_t*) Calloc(nnz+m + n+1, sizeof *cls_pool);

    lit_t **var = (lit_t**) Calloc(n+1, sizeof *var);
    lit_t **cls = (lit_t**) Calloc(m+1, sizeof *cls);
    int *cls_weight = (int*) Calloc(m+1, sizeof *cls_weight);

    int *var_pos = (int*) Calloc(n+1, sizeof *var_pos);
    int *cls_pos = (int*) Calloc(m+1, sizeof *cls_pos);
    var_pos[0] = 0, var_len[0] = m;
    cls_pos[0] = 0, cls_len[0] = n;
    for(int i=1; i<=n; i++) {
        var_pos[i] = var_pos[i-1] + var_len[i-1]+1;
        var[i] = var_pool + var_pos[i];
    }
    for(int j=1; j<=m; j++) {
        cls_pos[j] = cls_pos[j-1] + cls_len[j-1]+1;
        cls[j] = cls_pool + cls_pos[j];
    }

    // initialize truth vector
    var[0] = var_pool;
    for(int i=1; i<=m; i++) var_pool[i-1] = i;

    cls[0] = cls_pool;
    for(int j=1; j<=n; j++) cls_pool[j-1] = j;

    // initialize variables
    fseek(fin, 0, SEEK_SET);

    clause_cnt = 1;
    lineno = 1;
    for(; -1 != getline(&line, &linecap, fin); lineno++) {
        if(line[0] == '\0' || strchr("c\n", line[0]))
            continue;
        if(line[0] == 'p')
            continue;
        char *p = strtok(line, delim);
        if(has_weight){
            cls_weight[clause_cnt] = atoi(p);
            p = strtok(NULL, delim);
        }else{
            cls_weight[clause_cnt] = 1;
        }
        while(p){
            int literal = atoi(p);
            if(literal == 0) break;
            int sign = (literal>0)? 1 : -1;
            int var = abs(literal);
            var_pool[var_pos[var]++] = (clause_cnt<<1) | (sign>0);
            cls_pool[cls_pos[clause_cnt]++] = (var<<1) | (sign>0);

            p = strtok(NULL, delim);
        }
        clause_cnt++;
    }

    prob->m = m;
    prob->n = n;
    prob->nnz = nnz;
    prob->cls_len = cls_len;
    prob->var_len = var_len;
    prob->var = var;
    prob->cls = cls;
    prob->cls_weight = cls_weight;
    free(var_pos);
    free(cls_pos);
    free(line);
}

void szero(float *v, int l);
void saxpy(float *restrict y, float a, const float *restrict x, int l);
float snrm2(const float *x, int l);
void sscal(float *x, float a, int l);
float sdot(const float *x, const float *y, int l);
void scopy(float *x, float *y, int l);

void szero(float *v, int l)
{
    memset(v, 0, l*sizeof(*v));
}
void saxpy(float *restrict y, float a, const float *restrict x, int l)
{
    y = __builtin_assume_aligned(y, 4*sizeof(float));
    x = __builtin_assume_aligned(x, 4*sizeof(float));
    __m128 const a_ = _mm_set1_ps(a);
    for(int i=0; i<l; i+=4, x+=4, y+=4){
        __m128 y_ = _mm_load_ps(y);
        __m128 x_ = _mm_load_ps(x);
        y_ = _mm_add_ps(_mm_mul_ps(a_, x_), y_);
        _mm_store_ps(y, y_);
    }
}
float snrm2(const float *x, int l)
{
        float xx = sdot(x, x, l);
        return sqrt(xx);
}
void sscal(float *x, float a, int l)
{
        int m = l-4;
        int i;
        for (i = 0; i < m; i += 5){
                x[i] *= a;
                x[i+1] *= a;
                x[i+2] *= a;
                x[i+3] *= a;
                x[i+4] *= a;
        }

        for ( ; i < l; i++)        /* clean-up loop */
                x[i] *= a;
}
float sdot(const float *restrict x, const float *restrict y, int l)
{
    x = __builtin_assume_aligned(x, 4*sizeof(float));
    y = __builtin_assume_aligned(y, 4*sizeof(float));
    __m128 s = _mm_set1_ps(0);
    for(int i=0; i<l; i+=4, x+=4, y+=4){
        __m128 x_ = _mm_load_ps(x);
        __m128 y_ = _mm_load_ps(y);
        __m128 t = _mm_dp_ps(x_, y_, 0xf1);
        s = _mm_add_ss(s, t);
    }
    float s_;
    _mm_store_ss(&s_, s);

    return s_;
}
void scopy(float *x, float *y, int l)
{
        memcpy(y, x, sizeof(*x)*(size_t)l);
}

void rand_unit(float *x, int k)
{
        for(int i=0; i<k; i++) {
                x[i] = drand48()*2-1;
        }
        float r = snrm2(x, k);
        sscal(x, 1/r, k);
}

void rand_inertia(float *x, float *y, float **V, int n, int k)
{
    szero(x, k);
    for (int i=0; i<=n; i++) {
        saxpy(x, (drand48()*2-1)*y[i], V[i], k);
    }
    float r = snrm2(x, k);
    sscal(x, 1/r, k);
}

// A specialized version of saxpy with a being +1 or -1
// y += sign*x
void saxpy_bin(float *restrict y, float sign, float *restrict x, int k)
{
    __m128 const sign_ = _mm_set1_ps(0.*sign);
    y = __builtin_assume_aligned(y, 4*sizeof(float));
    x = __builtin_assume_aligned(x, 4*sizeof(float));
    for (int i=k; i; i-=4, x+=4, y+=4) {
        __m128 x_ = _mm_load_ps(x);
        __m128 y_ = _mm_load_ps(y);
        y_ = _mm_add_ps(_mm_xor_ps(x_, sign_), y_);
        _mm_store_ps(y, y_);
    }
}

float sasum(float *x, int k)
{
    float s = 0;
    for (int i=0; i<k; i++) s += fabs(x[i]);
    return s;
}

// Record traversal on binary tree
typedef struct path_t {
    int cap, len, ref_count;
    lit_t *pool, *rpivot, *curr;

    struct path_t *parent;
    lit_t node;
} path_t;

// path := index in path pool
// *pool := (parent path index) << 1 | assignment

path_t *path_init(path_t *parent, lit_t *rpivot)
{
    int cap = 8;
    path_t *self = Calloc(1, sizeof *self);
    *self = (path_t) {
        .cap = cap, .len = 1, .ref_count = 0,

        .pool = Calloc(cap, sizeof *self->pool),
        .rpivot = rpivot,

        .parent = parent, .node = parent?*parent->curr:0,
    };
    self->curr = self->pool;

    return self;
}

void path_free(path_t *self)
{
    free(self->pool);
    free(self);
}

int path_len(path_t *self, lit_t node)
{
    if (!self) return 0;

    lit_t *pool = self->pool;
    int len = 0;
    for (; node; node = pool[node/2-1], len++)
        ;
    return len;
}

void path_to_lit(path_t *src, lit_t node, lit_t *dst, int recursive)
{
    lit_t *orig_dst = dst, len;

    // fill in dst in reverse path order
    for (; src; node = src->node, src = src->parent) {
        lit_t *pool = src->pool;
        for (len=0; node; node = pool[node/2-1], len++)
            *dst++ = (node&1);
        for (lit_t *q=src->rpivot, *r=dst-1; len; len--, q--, r--) {
            *r |= *q & (~1u); 
        }
        if (!recursive) break;
    }

    // reverse
    dst--;
    for (lit_t t; orig_dst < dst; orig_dst++, dst--)
        t = *dst, *dst = *orig_dst, *orig_dst = t;
}

void path_move(path_t *self, int is_forward, lit_t b)
{
    if (is_forward) {
        if (self->cap == self->len) {
            lit_t *orig = self->pool;
            self->cap = ceil(self->cap*1.5);
            self->pool = realloc(self->pool, self->cap * sizeof *self->pool);
            assert(self->pool);
            self->curr = (self->curr-orig)+self->pool;
        }

        self->pool[self->len] = ((uint32_t)(self->curr - self->pool)+1)<<1 | (b&1);
        self->curr = &self->pool[self->len++];
    } else {
        self->curr = &self->pool[*self->curr/2-1];
    }
}

typedef struct {
    int n, curr;

    lit_t *pivot, **cls, ***watched;
    path_t *path;
} wstack_t;

typedef struct pair_t {
    lit_t lit;
    float val;
} pair_t;

typedef struct {
    int m, n, k, nnz;
    float **V, **Z, *y;

    float *primal, *loss, *clipped;
    float *dual, *s0, *delta;
    float *alpha, *zsqr, *b;
    pair_t **var;
    lit_t **fill, *perm, *order, *var_len;
    lit_t *cls_id, **cls_end;
} mixsat_t;

// p=watched[lit] is ptr stack top (with NULL barrier) containing ptr to cls lit
// the stack grows from higher address to lower address:  p p+1 .. 0 
// *p  = ptr to cls lit
// **p = lit value in cls

void forward(wstack_t *w, int b)
{
    int nfalse = 0;
    lit_t ***watched = w->watched;
    lit_t lit = (w->n-- << 1) | (b&1);
    for (lit_t **p=watched[lit^1]; *p; p++) {
        if (**p) { // append next watched lit to ptr
            *--watched[**p] = *p+1;
        } else {   // cls is false
            nfalse++;
        }
    }
    w->curr += nfalse;
}

void backtrack(wstack_t *w, int b)
{
    int nfalse = 0;
    lit_t ***watched= w->watched;
    lit_t lit = (++w->n << 1) | (b&1);
    for (lit_t **p=watched[lit^1]; *p; p++) {
        if (**p) { // if the previous lit in cls is non-zero
            watched[**p]++;
        } else {
            nfalse++;
        }
    }
    w->curr -= nfalse;
}

wstack_t *wstack_init(SATProblem *prob)
{
    wstack_t *self = Calloc(1, sizeof *self);
    int m = prob->m, n = prob->n;
    *self = (wstack_t) {
        .n = n, .curr = 0,

        .pivot = Calloc(n+1, sizeof *self->pivot),
        .cls = prob->cls,
        .watched = Calloc((n+1)*2, sizeof *self->watched),

        .path = NULL,
    };
    for (int i=1; i<=n; i++) self->pivot[i] = i<<1;

    // push all clauses into watched[2] for init (no ordering)
    self->watched[0] = Calloc(m+1+1, sizeof **self->watched);
    for (int i=1; i<(n+1)*2; i++) self->watched[i] = self->watched[0];
    lit_t **p = self->watched[2] = self->watched[0]+1;
    for (lit_t j=1; j<=m; j++, p++) *p = prob->cls[j]+1;

    return self;
}

void wstack_free(wstack_t *self)
{
    free(self->pivot);
    free(self->watched);
    free(self->cls[0]); free(self->cls);

    free(self);
}

// cmp for decreasing order
int cmp_by_lit(const void *a, const void *b)
{
    lit_t x = *(lit_t*)a, y = *(lit_t*)b;
    if(x<y) return 1;
    else if(x>y) return -1;
    else return 0;
}

float *SCORE;
// sort in increasing order
int cmp_by_score(const void *a, const void *b)
{
    float x = SCORE[*(int *)a], y = SCORE[*(int *)b];
    x = fabs(x), y = fabs(y);
    if (x>y) return 1;
    else if (x<y) return -1;
    else return 0;
}

wstack_t *wstack_commit(wstack_t *w, mixsat_t *mix)
{
    int n = mix->n, m = mix->m, nnz = mix->nnz;

    wstack_t *self = Calloc(1, sizeof *self);
    *self = (wstack_t) {
        .n = n, .curr = w->curr,

        .pivot =   Calloc(n+1, sizeof *self->pivot),
        .cls =     Calloc(m+1, sizeof *self->cls),
        .watched = Calloc((n+1)*2, sizeof *self->watched),
    };

    // decide order
    lit_t *perm = mix->perm, *order = mix->order;
    SCORE = mix->y;
    for (int i=1; i<=n; i++) perm[i] = i;
    qsort(perm+1, n, sizeof *perm, cmp_by_score); 
    for (int i=1; i<=n; i++) order[perm[i]] = i;

    // allocate watched stack
    lit_t ***watched = self->watched;
    watched[0] = Calloc(nnz+2*(n+1)+1, sizeof **watched);
    watched[1] = watched[0]+1;
    for (int t=2; t<(n+1)*2; t++) 
        watched[t] = watched[t-1] + mix->var_len[(perm[t>>1]<<1) | (t&1)] + 1;

    // generate clauses
    lit_t **cls = self->cls;
    lit_t *pcls = cls[0] = Calloc(nnz+m+1, sizeof **cls);

    for (lit_t **p=mix->fill, j=1; *p; p++, j++) {
        cls[j] = pcls;
        for (lit_t *q=*p; *q; q++) *pcls++ = (order[*q>>1]<<1) | (*q&1);
        qsort(cls[j], pcls-cls[j], sizeof(lit_t), cmp_by_lit);
        *pcls++ = 0;

        int is_redundant = 0;
        for (lit_t *q=cls[j]; *q; q++) {
            if ((*q ^ *(q+1)) == 1) { is_redundant = 1; break; }
        }

        if (is_redundant) fprintf(stderr, "is redundant\n");
        if (is_redundant) *cls[j] = 0, pcls = cls[j]+1;
        else *--watched[*cls[j]] = cls[j]+1;

        mix->cls_end[j] = pcls;
        for (lit_t *q=cls[j]; *q; q++) mix->cls_id[q-cls[0]+1] = j;
    }

    self->path = path_init(w->path, self->pivot+n);

    return self;
}

int maxsat_rounding(mixsat_t *mix, wstack_t *w, int n_trial, int *best, lit_t *best_sol, int64_t time_st)
{
    int n = mix->n, k = mix->k;
    float **V = mix->V, *r = mix->Z[0];

    lit_t *sol = Calloc(n+1, sizeof *sol);
    lit_t *perm = mix->perm, *pivot = w->pivot;

    int upper = mix->m+w->curr;
    for (int trial=0; trial<n_trial; trial++) {
        rand_unit(r, k);
        float wsign = sdot(r, V[0], k);
        int i;
        for(i=n; i>=1 && upper > w->curr; i--) {
            sol[i] = (wsign * sdot(r, V[perm[i]], k) > 0);
            forward(w, sol[i]);
        }

        if (upper > w->curr) {
            upper = w->curr;
            for (int j=1; j<=n; j++) pivot[j] = (pivot[j]& ~1u) | sol[j];
        }
        if (*best > w->curr) {
            *best = w->curr;
            printf("best %d time %fs\n", 
                    *best, wall_time_diff(wall_clock_ns(), time_st));
            fflush(stdout);
        }

        for (i+=1; i<=n; i++) {
            backtrack(w, sol[i]);
        }
        assert(w->n == n);
    }

    free(sol);

    return upper;
}

int mixsat_get_k(int n)
{
    int k = ceil(sqrt(n*2));
    if(k%4) k = (k+4)/4*4;

    return k;
}

mixsat_t *mixsat_init(SATProblem *prob)
{
    int m = prob->m, n = prob->n, k = mixsat_get_k(n);
    mixsat_t *self = Calloc(1, sizeof *self);
    *self = (mixsat_t) {
        .m = m, .n = prob->n, .k = k, .nnz = prob->nnz,

        .V =       Calloc(n+1, sizeof *self->V),
        .Z =       Calloc(m+1, sizeof *self->Z),
        .y =       Calloc(n+1, sizeof *self->y),
        .primal =  Calloc(n+1, sizeof *self->primal),
        .loss =    Calloc(m+1, sizeof *self->loss),
        .clipped = Calloc(n+1, sizeof *self->clipped),
        .dual =    Calloc(n+1, sizeof *self->dual),
        .s0  =     Calloc(m+1, sizeof *self->s0),
        .delta =   Calloc(n+1, sizeof *self->delta),

        .alpha =   Calloc(m+1, sizeof *self->alpha),
        .zsqr =    Calloc(m+1, sizeof *self->zsqr),
        .b =       Calloc(m+1, sizeof *self->b),

        .var =     Calloc(n+1, sizeof *self->var),

        .fill =    Calloc(m+1, sizeof *self->fill),
        .perm =    Calloc(n+1, sizeof *self->perm),
        .order =   Calloc(n+1, sizeof *self->order),
        .var_len = Calloc((n+1)*2, sizeof *self->var_len),

        .cls_id =  Calloc(prob->nnz+m+2, sizeof *self->cls_id),
        .cls_end = Calloc(m+1, sizeof *self->cls_end),
    };

    self->V[0] = aligned_alloc(4*sizeof(float), k*(n+1)* sizeof **self->V);
    self->Z[0] = aligned_alloc(4*sizeof(float), k*(m+1)* sizeof **self->Z);
    self->var[0] = Calloc(prob->nnz+n+1, sizeof **self->var);

    return self;
}

void mixsat_free(mixsat_t *self)
{
    free(self->V[0]); free(self->V);
    free(self->Z[0]); free(self->Z);
    free(self->var[0]); free(self->var);

    free(self->y);
    free(self->primal);
    free(self->loss);
    free(self->clipped);
    free(self->dual);
    free(self->s0);
    free(self->delta);
    free(self->fill);
    free(self->perm);
    free(self->var_len);

    free(self);
}

#define SIGN(x) ((int32_t)(((x)&1)<<1)-1)

void mixsat_select(mixsat_t *self, wstack_t *w)
{
    int m = 0, n = w->n;
    lit_t **fill = self->fill;
    memset(self->var_len, 0, 2*(n+1)*sizeof *self->var_len);
    for (int t=2; t<(n+1)*2; t++) {
        for (lit_t **p=w->watched[t]; *p; p++, m++) {
            *fill = *p-1;
            for (lit_t *q=*fill++; *q; q++) self->var_len[*q]++;
        }
    }
    *fill = NULL;
    self->m = m;
    self->n = n;

    self->nnz = 0;
    for (int t=2; t<(n+1)*2; t++) self->nnz += self->var_len[t];
}

void mixsat_setup(mixsat_t *self, wstack_t *w)
{
    // select cls
    mixsat_select(self, w);
    int n = w->n, m = self->m;
    for (int i=1; i<=n; i++) {
        int len = self->var_len[i*2] + self->var_len[i*2+1];
        self->var[i] = self->var[i-1] + len + 1;
        self->var[i]->lit = 0;
    }

    // init v and z
    float **V = self->V, **Z = self->Z;
    int k = mixsat_get_k(n); self->k = k;
    for (int i=1; i<=n; i++) V[i] = V[i-1] + k;
    for (int j=1; j<=m; j++) Z[j] = Z[j-1] + k;

    for (int i=0; i<=n; i++) rand_unit(V[i], k);
    memset(Z[0], 0, (m+1)*k * sizeof **Z);

    // fill in var
    for (lit_t j=1, **p=self->fill; *p; p++, j++) {
        saxpy_bin(Z[j], -1, V[0], k);

        int len = 0;
        for (lit_t *q=*p; *q; q++) len++;
        for (lit_t *q=*p; *q; q++) {
            int i = *q>>1;
            float sign = SIGN(*q);
            saxpy_bin(Z[j], sign, V[i], k);

            *--(self->var[i]) = (pair_t) {
                .lit = (j<<1) | (*q&1),
                .val = sign / (4*len),
            };
        }
    }
}

// consider the \min unsat problem,
float do_mixing(mixsat_t *mix, int max_iter, float eps)
{
    int n = mix->n, k = mix->k;

    float **V = mix->V, **Z = mix->Z, *y = mix->y;
    pair_t **var = mix->var;
    float *g = Z[0], diff = 0, prev_delta = mix->m;

    int iter = 0;
    for (; iter < max_iter; iter++) {
        float delta = 0;
        for (int i=1; i<=n; i++) {
            if(!var[i]->lit) continue;
            szero(g, k);
            float t = 0;
            for (pair_t *p=var[i]; p->lit; p++) {
                float sign = SIGN(p->lit);
                int j = p->lit >> 1;
                saxpy(g, p->val, Z[j], k);
                t -= sign*p->val;
            }

            saxpy(g, t, V[i], k);
            float gnrm = snrm2(g, k);
            y[i] = gnrm;

            if (gnrm < MEPS)  continue;

            sscal(g, -1/gnrm, k);
            delta += gnrm * (1 - sdot(g, V[i], k))*2;
            saxpy_bin(g, -1, V[i], k); // g  = g-vi
            saxpy_bin(V[i], 1, g, k);  // vi = vi+(g-vi)

            // Add vi back to Cj
            for (pair_t *p=var[i]; p->lit; p++) {
                float sign = SIGN(p->lit);
                int j = p->lit >> 1;
                saxpy_bin(Z[j], sign, g, k);
            }
        }
        diff -= delta;
        if (delta < 1e-10 || delta / (1-delta/prev_delta) < eps) break;
        prev_delta = delta;
    }

    if (iter==max_iter)
        fprintf(stderr, "Reach max iter (%d)!\n", iter);

    return diff;
}

float square(float x)
{   return x*x;   }

float loss(float znrm, int len)
{
    if (len==0) return 1;
    return (square(znrm)-square(len-1))/(len*4);
}

float positive(float x)
{
    if (x<=0) return 0;
    else return x;
}

float clip(float x)
{
    return (x>=0)*x + (x>=1)*(1-x);
}

float mixsat_eval(mixsat_t *mix, int curr)
{
    float fval = 0, clipped = 0;
    for (lit_t **p=mix->fill, j=1; *p; p++, j++) {
        lit_t len = 0;
        for (lit_t *q=*p; *q; q++, len++)
            ;
        mix->loss[j] = loss(snrm2(mix->Z[j], mix->k), len);
        fval += mix->loss[j];
        clipped += positive(mix->loss[j]);
    }
    mix->primal[0] = mix->dual[0] = fval+curr;
    mix->clipped[0] = clipped+curr;
    return mix->primal[0];
}

enum{BACKTRACK, FORWARD};
// only primal_move() -> dual_move() or dual_move() accepted
void primal_move(mixsat_t *mix, wstack_t *w, int is_forward, lit_t b)
{
    float diff = 0, cdiff = 0, **V = mix->V;
    int k = mix->k, i = mix->perm[w->n];
    lit_t lit = (w->n<<1) | (b&1);

    for (lit_t **p=w->watched[lit]; *p; p++) {
        int j = mix->cls_id[*p - w->cls[0]];
        diff -= mix->loss[j];
        cdiff -= positive(mix->loss[j]);
    }
    for (lit_t **p=w->watched[lit^1]; *p; p++) {
        int j = mix->cls_id[*p - w->cls[0]], len = mix->cls_end[j]-*p;
        float sign = SIGN(lit^!is_forward);

        diff -= mix->loss[j];
        cdiff -= positive(mix->loss[j]);
        saxpy_bin(mix->Z[j], sign, V[i], k);
        mix->loss[j] = loss(snrm2(mix->Z[j], k), len-is_forward);
        diff += mix->loss[j];
        cdiff += positive(mix->loss[j]);
    }

    if(is_forward) mix->primal[1] = mix->primal[0]+diff;
    if(is_forward) mix->clipped[1] = mix->clipped[0]+cdiff;
    mix->primal += SIGN(is_forward);
    mix->clipped += SIGN(is_forward);
}

void dual_move(mixsat_t *mix, wstack_t *w, int is_forward, lit_t b)
{
    float *delta = mix->delta;
    int i = mix->perm[w->n];

    float diff = +2*sasum(delta, mix->n+1) + mix->y[i];
    for (pair_t *p = mix->var[i]; p->lit; p++) {
        // constant
        int j = p->lit>>1;
        float *s0j = &mix->s0[j];
        diff -= square(*s0j-1) * fabs(p->val);
        *s0j += SIGN(p->lit ^ b ^ is_forward);
        diff += square(*s0j-1) * fabs(p->val);
        diff -= fabs(p->val);

        assert(!is_forward || (*w->cls[j]&1) == (p->lit&1));
        // delta
        for (lit_t *q = w->cls[j]+is_forward; *q; q++) {
            mix->delta[*q>>1] += SIGN(*q ^ p->lit ^ b ^ is_forward) * fabs(p->val);
            assert((*q>>1)<w->n);
        }
        w->cls[j] += SIGN(is_forward);
    }
    diff -= 2*sasum(delta, w->n);

    if (is_forward) mix->dual[1] = mix->dual[0]+diff;
    mix->dual += SIGN(is_forward);
}

void wstack_move(wstack_t *self, lit_t node, int is_forward)
{
    int len = path_len(self->path, node);
    lit_t *assign = Calloc(len+1, sizeof *assign);
    path_to_lit(self->path, node, assign, 0);

    if (is_forward) {
        for (int i=0; i<len; i++) forward(self, assign[i]);
    } else {
        for (int i=len-1; i>=0; i--) backtrack(self, assign[i]);
    }
    free(assign);
}

// the heap element
typedef struct heapnode_t {
    float val;
    lit_t node;
    wstack_t *parent;
} heapnode_t;

// the heap
typedef struct heap_t {
    int len, cap;
    heapnode_t *root;
} heap_t;

heap_t *heap_init()
{
    int cap = 1024;
    heap_t *self = Calloc(1, sizeof *self);
    *self = (heap_t) {
        .len = 0, .cap = cap,
        .root = Malloc(cap * sizeof(heapnode_t)),
    };
    return self;
}

void heap_free(heap_t *h)
{
    free(h->root);
    free(h);
}

heapnode_t heap_pop(heap_t *h)
{
    heapnode_t *root = h->root;
    heapnode_t ret = root[0];
    heapnode_t p = root[--h->len];
    int i, c;
    for (i=0; (c=i*2+1) < h->len; i = c) {
        if (c+1 < h->len && root[c].val > root[c+1].val) c++;
        if (p.val < root[c].val) break;
        else root[i] = root[c];
    }
    root[i] = p;
    return ret;
}

void heap_push(heap_t *h, float val, lit_t node, wstack_t *parent)
{
    if (h->cap == h->len) {
        h->cap = ceil(h->cap*1.5);
        h->root = realloc(h->root, h->cap * sizeof *h->root);
        assert(h->root);
    }
    int i = h->len++, c;
    heapnode_t *root = h->root;
    for(; i && root[(c=(i-1)/2)].val > val; i=c)
        root[i] = root[c];
    root[i] = (heapnode_t){.val = val, .node = node, .parent = parent};
}

typedef struct result_t {
    int best;
    lit_t *best_sol;

    int n_visited, n_expanded, n_pruned;
    int64_t time_st;
} result_t;

void print_sol(lit_t *p, int n)
{
    for (int i=0; i<n; i++, p++)
        fprintf(stderr, "%c%d ", (*p&1)?'+':'-', *p>>1);
    fprintf(stderr, "\n");
}

void print_path(wstack_t *self, lit_t node)
{
    if(!self->path) return; 
    int len = path_len(self->path, node);
    lit_t *assign = Calloc(len, sizeof *assign);
    path_to_lit(self->path, node, assign, 0);
    print_sol(assign, len);
    free(assign);
}

int bin_maxsat(SATProblem prob, Parameter *param)
{
    int m = prob.m, n = prob.n;

    wstack_t *w = wstack_init(&prob);
    mixsat_t *mix = mixsat_init(&prob);

    int64_t time_st = wall_clock_ns();
    int best = m;
    lit_t *best_sol = Calloc(n+1, sizeof *best_sol);
    int64_t n_visited = 0, n_pruned = 0, n_expanded = 0;

    heap_t *Q = heap_init();
    heap_push(Q, 0, 0, w);
    for (int n_searched=0; Q->len != 0; n_searched++) {
        heapnode_t h = heap_pop(Q);
        wstack_move(h.parent, h.node, FORWARD);

        mixsat_setup(mix, h.parent);
        do_mixing(mix, param->max_iter, param->eps);
        float val = mixsat_eval(mix, h.parent->curr);
        if (ceil(val-param->eps) >= best) {
            n_pruned++;
            wstack_move(h.parent, h.node, BACKTRACK); 
            continue; 
        }

        wstack_t *w = wstack_commit(h.parent, mix);
        wstack_move(h.parent, h.node, BACKTRACK); 
        maxsat_rounding(mix, w, sqrt(w->n)*(w->n==n?500:1), &best, best_sol, time_st);

        for (int from_parent=1, depth=1; depth; n_visited++) {
            int action;
            if (!from_parent) {
                lit_t b = *w->path->curr & 1;
                backtrack(w, b);
                path_move(w->path, BACKTRACK, b);
                dual_move(mix, w, BACKTRACK, b);
                primal_move(mix, w, BACKTRACK, b);

                action = b == (w->pivot[w->n]&1)
                        && *w->watched[w->n*2+(b^1)] ;
            } else if (ceil(*mix->dual) >= best || w->curr >= best) {// prunable
                action = BACKTRACK, n_pruned++;
            } else if (*mix->primal < best && depth<=8) { // unprunable
                action = FORWARD, n_expanded++; 
            } else { // uncertain
                heap_push(Q, *mix->clipped, *w->path->curr, w);
                w->path->ref_count++;
                action = BACKTRACK;
            }

            if (0 && from_parent) fprintf(stderr, "depth %d parent %d primal %f dual %f action %d n %d top %f best %d\n",
                    depth, from_parent, *mix->primal, *mix->dual, action, w->n, Q->root->val, best);

            if (action) { // going down
                lit_t b = w->pivot[w->n]&1;
                lit_t next = b ^ (!from_parent || !*w->watched[w->n*2+b]);
                primal_move(mix, w, FORWARD, next);
                dual_move(mix, w, FORWARD, next);
                path_move(w->path, FORWARD, next);
                forward(w, next);

                from_parent = 1, depth++;
            } else { // going up
                from_parent = 0, depth--;
            }
        }
    }

    printf("best %d visited %u pruned %u expand %u mix %u time %.2fs\n",
            best, (int)n_visited, (int)n_pruned, (int)n_expanded, (int)(n_pruned+n_expanded),
                    wall_time_diff(wall_clock_ns(), time_st)
            );
    for(lit_t *p=best_sol+1; p<=best_sol+n; p++)
        printf("%c%u ", (*p&1)?'+':'-', *p>>1);
    printf("\n");
    return best;
}

void print_usage(char* prog_name, Parameter *param)
{
    printf( "%s [OPTIONS] INPUT: Mixing method for SDP\n", prog_name); 
    printf( "OPTIONS:\n");
    printf( "\t-k RANK: rank of solution (default auto)\n");
    printf( "\t         use \"-k /2\" to divide the rank by 2\n");
    printf( "\t-e EPS: stopping threshold (default %.4e)\n", param->eps);
    printf( "\t-t MAX_ITER: maximum iteration (default %d)\n", param->max_iter);
    printf( "\t             use \"-t max\" for INT_MAX\n");
    printf( "\t-r N_TRIAL: number of trial in evaluation (default %d)\n", param->n_trial);
    printf( "\t-v: verbose\n");
}

void get_parameter(int argc, char **argv, Parameter *param)
{
    Parameter _param = {
        .solver = MAXSAT,
		.k = -1,
        .eps = 2e-2,
        .max_iter = 100,
        .verbose = 0,
        .n_trial = 1,
        .fin = NULL,
        .fin_name = NULL,
    };

    if(argc <= 1){
        print_usage(argv[0], &_param);
        exit(0);
    }

    char **p = argv+1;
    int i;
    for(i=1; i<argc; i++, p++){
        if(!strcmp(*p, "-k")){
            if(i+1  >= argc) break;
            int ret = sscanf(p[1], "/%d", &_param.k);
            if(ret==1){
                _param.k *= -1;
            }else{
                ret = sscanf(p[1], "%d", &_param.k);
                if(ret != 1 || _param.k <= 0) break;
            }
            i++, p++;
        }else if(!strcmp(*p, "-e")){
            if(i+1 >= argc) break; 
            int ret = sscanf(p[1], "%lf", &_param.eps);
            if(ret != 1) break; 
            i++, p++;
        }else if(!strcmp(*p, "-t")){
            if(i+1 >= argc) break; 
            if(!strcmp(p[1], "max")){
                _param.max_iter = INT_MAX;
            }else{
                int ret = sscanf(p[1], "%d", &_param.max_iter);
                if(ret != 1) break;
            }
            i++, p++;
        }else if(!strcmp(*p, "-r")){
            if(i+1 >= argc) break;
            int ret = sscanf(p[1], "%d", &_param.n_trial);
            if(ret != 1) break; 
            if(_param.n_trial < 1)
                _param.n_trial = 1;
            i++, p++;
        }else if(!strcmp(*p, "-v")){
            _param.verbose = 1;
        }else if(i+1 == argc){
            _param.fin = fopen(*p, "r");
            if(!_param.fin){
                fprintf(stderr, "%s\n", strerror(errno));
                exit(1);
            }
            _param.fin_name = strdup(*p);
        }else{
            printf("Error: no such parameter\n");
            break;
        }
    }
    if(i != argc || !_param.fin){
        print_usage(argv[0], &_param);
        exit(0);
    }
    *param = _param;
}

int main(int argc, char **argv)
{
    Parameter param;
    get_parameter(argc, argv, &param);
    srand48(0);

    SATProblem sat_prob;
    printf("Solving maximum SAT. Reading %s\n", param.fin_name);
    read_sat_problem(param.fin, &sat_prob);
    printf("m %d n %d nnz %d\n", sat_prob.m, sat_prob.n, sat_prob.nnz);

    bin_maxsat(sat_prob, &param);
    return 0;
}
