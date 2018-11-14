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

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
    // init
    if(max_line_len == 0) {
        max_line_len = 1024;
	    line = (char *)Malloc(max_line_len*sizeof(char));
    }

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

int is_int(char *s)
{
    if(!s || *s == '\0')
        return 0;
    if(*s!='-' && !isdigit(*s)) 
        return 0;
    s++;
    for(; *s; s++) {
        if(!isdigit(*s))
            return 0;
    }
    return 1;
}

int is_valid_literal(char *s, int nvar)
{
    if(!is_int(s)) return 0;
    int var = abs(atoi(s));
    if(var != 0 && var <= nvar)
        return 1;
    return 0;
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
    for(; readline(fin); lineno++) {
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
    for(; readline(fin); lineno++) {
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

// z += sign*v, g += beta*z
void mix_update(float *restrict g, float sign, float beta, float *restrict z, float *restrict v, int k)
{
    __m128 const beta_ = _mm_set1_ps(beta);
    __m128 const sign_ = _mm_set1_ps(0.*sign); // xor sign bit using minus zero
    g = __builtin_assume_aligned(g, 4*sizeof(float));
    z = __builtin_assume_aligned(z, 4*sizeof(float));
    v = __builtin_assume_aligned(v, 4*sizeof(float));
    for (int i=k; i; i-=4, g+=4, z+=4, v+=4) {
        __m128 v_ = _mm_load_ps(v);
        __m128 z_ = _mm_load_ps(z);
        __m128 g_ = _mm_load_ps(g);
        z_ = _mm_add_ps(_mm_xor_ps(v_, sign_), z_);
        g_ = _mm_add_ps(_mm_mul_ps(beta_, z_), g_);
        _mm_store_ps(z, z_);
        _mm_store_ps(g, g_);
    }
}

// y += sign*x
void scaddsub(float *restrict y, float sign, float *restrict x, int k)
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

typedef struct {
    int m, n;
    lit_t *pivot;

    lit_t ***watched, ***left;
    lit_t **cls;
    int *cls_id;
    lit_t **cls_end;

} wstack_t;

lit_t cls_lit(wstack_t *w, lit_t **p)
{
    return w->cls_id[(*p-w->cls[0])-1]<<1 | (*(*p-1)&1);
}

int cls_len(wstack_t *w, lit_t **p)
{
    int j = w->cls_id[(*p-w->cls[0])-1];
    return w->cls_end[j]-*p+1;
}

int *ORDER;

// p=watched[lit] is ptr stack top (with NULL barrier) containing ptr to cls lit
// the stack grows from higher address to lower address:  p p+1 .. 0 
// *p  = ptr to cls lit
// **p = lit value in cls

int forward(wstack_t *w, lit_t lit)
{
    int nfalse = 0;
    lit_t ***watched = w->watched;
    for (lit_t **p=watched[lit^1]; *p; p++) {
        lit_t r = **p, t=!r;
        *(watched[r]-1) = *p+1;
        nfalse += t;
        watched[r] -= !t;
    }
    return nfalse;
}

int backtrack(wstack_t *w, lit_t lit)
{
    int nfalse = 0;
    lit_t ***watched= w->watched;
    for (lit_t **p=watched[lit^1]; *p; p++) {
        lit_t r =  **p, t=!r;
        nfalse += t;
        watched[r] += !t;
    }
    return nfalse;
}

// cmp for decreasing order
int cmp_by_order(const void *a, const void *b)
{
    int x = ORDER[(*(lit_t*)a)>>1], y = ORDER[(*(lit_t*)b)>>1];
    if(x<y) return 1;
    else if(x>y) return -1;
    else return 0;
}

// cmp for increasing order of previous lit in cls
int cmp_by_preceeding_order(const void *a, const void *b)
{
    int x = ORDER[(*(*(lit_t**)a-1))>>1], y = ORDER[(*(*(lit_t**)b-1))>>1];
    if(x>y) return 1;
    else if(x<y) return -1;
    else return 0;
}

lit_t cls_lit2(wstack_t *w, lit_t *p)
{
    return w->cls_id[(p-w->cls[0])-1];
}
wstack_t *GW;
int cmp_by_cls_order(const void *a, const void *b)
{
    int x = cls_lit2(GW, *(lit_t**)a), y = cls_lit2(GW, *(lit_t**)b);
    if(x<y) return 1;
    else if(x>y) return -1;
    else return 0;
}

wstack_t wstack_init(SATProblem *prob)
{
    int m = prob->m, n = prob->n;

    wstack_t self = {
        .m = m, .n = n,

        .pivot =   Calloc(2*(n+1), sizeof *self.pivot),
        .watched = Calloc(2*(n+1), sizeof *self.watched),
        .left =    Calloc(2*(n+1), sizeof *self.left),
        .cls = prob->cls,
        .cls_id =  Calloc(prob->nnz+n+m+1, sizeof *self.cls_id),
        .cls_end = Calloc(m+1, sizeof *self.cls_end),
    };

    for (int i=1; i<=n; i++) self.pivot[i] = i<<1;
    ORDER = Calloc(n+1, sizeof *ORDER);
    for(int i=1; i<=n; i++) ORDER[i] = i;
    ORDER[0] = n+1;

    lit_t **cls = prob->cls;
    lit_t ***right = self.watched, ***left = self.left;

    int *cap = Calloc(2*(n+1), sizeof *cap);
    for (int j=1; j<=m; j++) {
        lit_t *p = cls[j];
        for (; *p; p++)
            cap[*p]++, self.cls_id[p-cls[0]] = j;
        self.cls_end[j] = p;
    }

    right[0] = Calloc(prob->nnz+2*n+2+1, sizeof **right);
    right[0]++;
    right[1] = right[0]+1;
    for (lit_t lit=1<<1; lit < (n+1)<<1; lit++) {
        right[lit] = right[lit-1] + cap[lit] + 1;
        left[lit] = right[lit-1];
    }
    for (int j=1; j<=m; j++) {
        qsort(cls[j], prob->cls_len[j], sizeof **cls, cmp_by_order);
        // remove redundant cls
        lit_t prev = 0;
        lit_t *p = cls[j];
        for(; *p && ((*p|1) != (prev|1)); prev=*p, p++)
            ;
        if(*p) continue;
        *--right[*cls[j]] = cls[j]+1;
    }



    free(cap);
    return self;
}

void wstack_update(wstack_t *w, int n)
{
    lit_t *pivot = w->pivot;
    lit_t ***left = w->left, ***right = w->watched;
    for (int i=1; i<=n; i++) ORDER[pivot[i]>>1] = i;

    for (int t=2; t<(n+1)<<1; t++) {
        lit_t lit = pivot[t>>1] ^ (t&1), **p;
        for (p=right[lit]; *p; p++) { // sort every owned cls
            int qlen = 1;
            for(lit_t *q=*p; *q; q++) qlen++;
            qsort(*p-1, qlen, sizeof **p, cmp_by_order);
        }
        for (p=right[lit]; *p; p++) { // distribute new cls head to left stack
            *++left[*(*p-1)] = *p-1;
        }
        right[lit] = p;
    }

    for (int t=2; t < (n+1)<<1; t++) {
        lit_t lit = pivot[t>>1] ^ (t&1), **p;
        int qlen = 0;
        for (p=left[lit]; *p; p--) *--right[lit] = *p, qlen++;
        left[lit] = p;
        qsort(right[lit], qlen, sizeof **right, cmp_by_preceeding_order);
        for (p=right[lit]; *p; p++) *p += 1;
    }

    for (int t=2; t<(n+1)<<1; t++) {
        lit_t lit = pivot[t>>1] ^ (t&1);
        for (lit_t **p=right[lit]; *p; p++) {
            int prev = n+1;
            for(lit_t *q=*p-1; *q; q++) {
                int j =  ORDER[*q>>1];
                if(prev <= j) fprintf(stderr, "prev %d j %d *q %d\n", prev, j, *q);
                assert(prev > j);
                prev = j;
            }
        }
    }
}

typedef struct sol_t {
    lit_t *answer;
    union{
        float f;
        int x;
    };
    int lv;
} sol_t ;

sol_t sol_init(int n)
{
    sol_t self = {
        .answer = Calloc(n+1, sizeof *self.answer),
    };
    return self;
}

void sol_copy(sol_t *src, sol_t *dst, int n, int lv, float val)
{
    memcpy(dst->answer, src->answer, n*sizeof *src->answer);
    dst->lv = lv;
    dst->f = val;
}

typedef struct {
    float **curr, **local, **orig;
    int *pos, len, cap, k, committed;
} overlay_t;

void overlay_push(overlay_t *self, int j)
{
    self->pos[self->len++] = j;
    assert(self->len <= self->cap);
}

void overlay_free(overlay_t *self)
{
    if (self->committed) {
        for (int i=0; i<self->len; i++)
            self->curr[self->pos[i]] = self->orig[i];
    }

    free(*self->local);
    free(self->local);
    free(self->orig);
    free(self->pos);
}

void overlay_reset(overlay_t *self, int cap, int k)
{
    float *p = self->local[0];
    memset(self->pos, 0, cap * sizeof *self->pos);
    memset(p, 0, cap*k * sizeof *p);
    for (int i=0; i<cap; i++, p+=k) self->local[i] = p;
    self->len = 0, self->cap = cap, self->k = k, self->committed = 0;
}

overlay_t overlay_init(float **curr, int cap, int k)
{
    overlay_t self = {
        .curr = curr,
        .orig =   Calloc(cap, sizeof *self.orig),
        .local =  Calloc(cap, sizeof *self.local),
        .pos =    Calloc(cap, sizeof *self.pos),
        .len = 0, .cap = cap, .k = k, .committed = 0,
    };

    float *p = aligned_alloc(4*sizeof(float), k*cap * sizeof *p);
    memset(p, 0, k*cap * sizeof *p);
    for (int i=0; i<cap; i++, p+=k) self.local[i] = p;

    return self;
}

void overlay_commit(overlay_t *self, overlay_t *dst)
{
    *dst = overlay_init(self->curr, self->len, self->k);
    memcpy(dst->local[0], self->local[0], self->len * self->k * sizeof(float));
    memcpy(dst->orig,     self->orig,     self->len * sizeof *self->orig);
    memcpy(dst->pos,      self->pos,      self->len * sizeof *self->pos);

    for (int i=0; i<self->len; i++) {
        int j = dst->pos[i];
        dst->orig[i] = self->curr[j];
        dst->curr[j] = dst->local[i];
    }
    dst->len = dst->cap = self->len;
    dst->k = self->k, dst->committed = 1;
}


typedef struct pair_t {
    lit_t lit;
    float val;
} pair_t;

typedef struct ref_t {
    lit_t *assign;
    int lv;
    int val;
} ref_t;

typedef struct {
    int max_iter;
    float eps;

    pair_t **var, **pool;
    overlay_t v, z, *ov, *oz;
    float **y;

    lit_t **fill, **pivots;
    float **dual;
} mixsat_t;

float square(float x)
{   return x*x;   }

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

void save_best(sol_t curr, sol_t *best, int n, int64_t time_st)
{
    memcpy(best->answer, curr.answer, (n+1)*sizeof *curr.answer);
    best->x= curr.x;
    printf("update : best %d at %.3f s\n", best->x, wall_time_diff(wall_clock_ns(),time_st));
    fflush(stdout);
}

int maxsat_rounding(mixsat_t *mix, wstack_t *w, int n_trial, int upper, sol_t *out, sol_t *best, int64_t time_st)
{
    lit_t *pivot = w->pivot;

    int n = mix->v.len-1, k = mix->v.k;
    float **V = mix->v.local;

    int *perm = Calloc(n+1, sizeof *perm);
    lit_t *sol = Calloc(n+1, sizeof *sol);
    float *r = mix->z.local[0];

    int gotcha =0;
    int curr = 0;

    upper = mix->z.len-1;
    for(int trial=0; trial<n_trial; trial++){
        rand_unit(r, k);
        float wsign = sdot(r, V[0], k);
        int i;
        for(i=n; i>=1 && curr < upper; i--) {
            sol[i] = (pivot[i] &~1u) | (wsign * sdot(r, V[i], k) > 0);
            curr += forward(w, sol[i]);
        }

        // find the tightest upper bound
        if(upper > curr){
            upper = curr, gotcha=1;
            for(int j=1; j<=n; j++) out->answer[j] = sol[j];
            if (best->x > out->x+upper) {
                out->x += upper;
                save_best(*out, best, ORDER[0]-1, time_st);
                out->x -= upper;
            }
        }

        for(i+=1; i<=n; i++) {
            curr -= backtrack(w, sol[i]);
        }
        assert(curr == 0);
    }

    if (gotcha) {
        for(int i=1; i<=n; i++) perm[i] = i;
        SCORE = *mix->y;
        //for(int i=1; i<=n; i++) SCORE[i] = -fabs(sdot(V[i], V[0], k));
        qsort(perm+1, n, sizeof *perm, cmp_by_score); 
        for(int i=1; i<=n; i++) sol[i] = out->answer[perm[i]];
        for(int i=1; i<=n; i++) out->answer[i] = sol[i];
    }

    free(perm);
    free(sol);
    return upper;
}

int mixsat_get_k(int n)
{
    int k = ceil(sqrt(n*2)); 
    if(k%4) k = (k+4)/4*4;

    return k;
}

mixsat_t mixsat_init(SATProblem *prob)
{
    int m = prob->m, n = prob->n, k = mixsat_get_k(n);
    mixsat_t self = {
        .max_iter = 50,
        .eps = 1e-1,

        .var =     Calloc(n+1, sizeof *self.var),
        .pool =    Calloc(n+1, sizeof *self.pool),
        .y =       Calloc(n+1, sizeof *self.y),
        .v =       overlay_init(Calloc(n+1, sizeof *self.v.curr), n+1, k),
        .z =       overlay_init(Calloc(2*(m+1), sizeof *self.z.curr), 2*(m+1), k),
        .ov =      Calloc(n+1, sizeof *self.ov),
        .oz =      Calloc(n+1, sizeof *self.oz),

        .fill     = Calloc(2*m+1, sizeof *self.fill),
        .pivots   = Calloc(n+1, sizeof *self.pivots),
        .dual     = Calloc(n+1, sizeof *self.dual),
    };

    self.pool[0] = Calloc(prob->nnz*2+n+1, sizeof **self.pool);
    for (int i=1; i<=n; i++)
        self.pool[i] = self.pool[i-1] + prob->var_len[i]*2 + 1;
    *self.y = Calloc(n+1, sizeof **self.y);

    return self;
}

void mixsat_free(mixsat_t *self)
{
    free(self->v.curr);
    free(self->z.curr);
    free(self->var);
    free(*self->pool);
    free(self->pool);
    free(*self->y);
    free(self->y);
    free(self->ov);
    free(self->oz);
}

void mixsat_push(mixsat_t *self, wstack_t *w, int lv, lit_t *assign)
{
    overlay_commit(&self->v, ++self->ov);
    overlay_commit(&self->z, ++self->oz);

    *++self->pivots = Calloc(lv+1, sizeof *assign);
    for (int i=1; i<=lv; i++)
        (*self->pivots)[i] = w->pivot[i], w->pivot[i] = assign[i];
    wstack_update(w, lv);

    *++self->y = Calloc(lv+1, sizeof **self->y);
    *++self->dual = Calloc(lv+1, sizeof **self->dual);
}

void mixsat_pop(mixsat_t *self, wstack_t *w)
{
    int lv = self->ov->len-1;
    for(int i=1; i<=lv; i++) w->pivot[i] = (*self->pivots)[i];
    wstack_update(w, lv);

    overlay_free(self->ov--);
    overlay_free(self->oz--);
    free(*self->pivots--);
    free(*self->y--);
    free(*self->dual--);
}

int wstack_cd(wstack_t *w, lit_t *src, int src_lv, lit_t *dst, int dst_lv, int common_lv)
{
    int diff = 0;
    for(int i=src_lv+1; i<=common_lv; i++) diff -= backtrack(w, src[i]);
    for(int i=common_lv; i>dst_lv; i--) diff += forward(w, dst[i]);
    return diff;
}

void squeeze_init(float *dst, float *src, int dst_k)
{
    float r = snrm2(src, dst_k);
    if (r<1e-1) rand_unit(dst, dst_k);
    else scopy(src, dst, dst_k), sscal(dst, 1/r, dst_k);
}

#define SIGN(x) ((int32_t)(((x)&1)<<1)-1)
#define SRC_ORDER(q) (ORDER[*(q-2)>>1])
#define LIT_ORDER(q) (ORDER[*(q-1)>>1])

float mixsat_set_var(mixsat_t *self, wstack_t *w, int n, int is_ref, lit_t *assign, int curr)
{
    // clean var
    for (int i=1; i<self->v.cap; i++)
        for (pair_t **p=&self->var[i]; *p && (*p)->lit; (*p)++)
            ;
    // select cls
    lit_t **fill = self->fill;
    int m = 0, nr = 0, top = is_ref?n+1:ORDER[0];
    for (int t=2; t < (n+1)<<1; t++) {
        lit_t lit = w->pivot[t>>1] ^ (t&1);
        for (lit_t **p=w->watched[lit]; *p && SRC_ORDER(*p)<=top; p++, m++) {
            *++fill = *p;
        }
    }
    float fval = 0;
    if (is_ref) {
        wstack_cd(w, assign, n, w->pivot, n, n+1);
        for (int t=2; t < (n+1)<<1; t++) {
            lit_t lit = w->pivot[t>>1] ^ (t&1);
            for (lit_t **p=w->watched[lit]; *p && SRC_ORDER(*p)<=top; p++, nr++) {
                *++fill = *p;
            }
        }
        float cdiff = wstack_cd(w, w->pivot, n, assign, n, n+1);
        fval += cdiff;
        m += nr;
    }

    // init v and z
    int k = mixsat_get_k(n);
    overlay_reset(&self->v, n+1, k);
    overlay_reset(&self->z, m+1, k);
    float **V = self->v.local, **Z = self->z.local;

    overlay_push(&self->v, 0);
    overlay_push(&self->z, 0);
    rand_unit(V[0], k);
    for (int i=1; i<=n; i++) {
        self->var[i] = self->pool[w->pivot[i]>>1];
        overlay_push(&self->v, w->pivot[i]>>1);
        if (n+1==ORDER[0]) rand_unit(V[i], k);
        else squeeze_init(V[i], self->ov->curr[w->pivot[i]>>1], k);
    }

    int range = is_ref?(n+1):ORDER[0];
    // fill in var
    int j = 1;
    for (lit_t **p=fill; *p; p--, j++) {
        float dir = SIGN(j>nr);
        overlay_push(&self->z, cls_lit(w,p)>>1);
        scaddsub(Z[j], -1, V[0], k);
        int len = cls_len(w, p);
        for (lit_t *q=*p-1; *q; q++) {
            int i = ORDER[*q>>1];
            assert(i<=range);
            float sign = SIGN(*q);
            scaddsub(Z[j], sign, V[i], k);
            if(i>n) continue;
            *--(self->var[i]) = (pair_t) {
                .lit = (j<<1) | (*q&1),
                .val = sign / (4*len) * dir,
            };
        }
        float loss = (square(snrm2(Z[j], k)) - square(len-1)) / (4*len);
        fval += loss * dir;
    }

    if(is_ref && 0)fprintf(stderr, "\nnew %c%d[%d]\n", (assign[n+1]&1)?'+':'-', w->pivot[n+1]>>1, n+1);

    if(is_ref && 0) {
        fprintf(stderr, "n %d m %d nr %d parent %d\n", n, m, nr, range);
        for (int i=1; i<=n; i++) {
            fprintf(stderr, "%d: ", i);
            for (pair_t *p=self->var[i]; p->lit; p++) {
                fprintf(stderr, "%c%d(%d) ", (p->lit&1)?'+':'-', p->lit>>1, abs((int)(1/p->val/4)));
            }
            fprintf(stderr, "\n");
        }
    }

    return fval;
}

// consider the \min unsat problem,
float do_mixing(mixsat_t *mix)
{
    int n = mix->v.len-1, k = mix->v.k;

    float **V = mix->v.local, **Z = mix->z.local, *y = *mix->y;
    pair_t **var = mix->var;
    float *g = mix->z.local[0], diff = 0, prev_delta=mix->z.cap-1;

    //int64_t time_st = wall_clock_ns();
    int iter = 0;
    for (; iter < mix->max_iter; iter++) {
        float delta = 0;
        for (int i=n; i>=1; i--) {
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

            if (gnrm < MEPS) { continue;
            } else {
                sscal(g, -1/gnrm, k);
                delta += gnrm * (1 - sdot(g, V[i], k))*2;
                scaddsub(g, -1, V[i], k); // g  = g-vi
                scaddsub(V[i], 1, g, k);  // vi = vi+(g-vi)
            }

            // Add vi back to Cj
            for (pair_t *p=var[i]; p->lit; p++) {
                float sign = SIGN(p->lit);
                int j = p->lit >> 1;
                scaddsub(Z[j], sign, g, k);
            }
        }
        diff -= delta;
        if (delta < 1e-10 || delta / (1-delta/prev_delta) < mix->eps) break;
        prev_delta = delta;
    }

    if (iter==mix->max_iter)
        fprintf(stderr, "Reach max iter (%d)!\n", iter);

    return diff;
}

float loss(float znrm, int len)
{
    if(len==0) return 1;
    return (square(znrm)-square(len-1))/(len*4);
}

float validate(wstack_t *w, mixsat_t *mix, int n)
{
    int k = mix->ov->k;
    float *z = aligned_alloc(4*sizeof(float), k * sizeof(float));
    float **V = mix->ov->curr, **Z = mix->oz->curr;
    float fval = 0;

    assert(V[0] == mix->ov->local[0]);
    int m = 1;
    for(int i=2; i<(n+1)*2; i++){
        lit_t lit = w->pivot[i>>1] ^ (i&1);
        assert(V[lit>>1] == mix->ov->local[ORDER[lit>>1]]);
        for(lit_t **p = w->watched[lit]; *p; p++, m++) {
            szero(z, k);
            scaddsub(z, -1, V[0], k);
            for(lit_t *q = *p-1; *q; q++)
                scaddsub(z, SIGN(*q), V[(*q)>>1], k);
            int len = cls_len(w, p);
            fval += loss(snrm2(z, k), len);
            scaddsub(z, -1, Z[cls_lit(w,p)>>1], k);
            float err = snrm2(z, k);
            if(err >= 1e-1){
                fprintf(stderr, "cls %d err = %f\n", cls_lit(w,p)>>1, err);
                lit_t *q = *p;
                for(; *q; q--);
                q++;
                for(; *q; q++)
                    fprintf(stderr, "%c%d order %d\n", (*q&1)?'+':'-', (*q)>>1, ORDER[(*q)>>1]);
            }
            assert(Z[cls_lit(w,p)>>1] == mix->oz->local[m]);
            assert(err < 1e-1);
        }
    }
    free(z);
    return fval;
}

#define SQRT5 2.23606797749979
float mix_forward(wstack_t *w, mixsat_t *mix, lit_t lit, int lv)
{
    float diff = 0, **V = mix->ov->curr, **Z = mix->oz->curr;
    int k = mix->ov->k;
    // for p in lit and lit^1
    mix->dual[0][lv-1] = mix->dual[0][lv] + mix->y[-1][lv]*2;
    for(lit_t **p=w->watched[lit], pos=1; *p||pos; p++){
        if (!*p) {p=w->watched[lit^1]-1; pos=0; continue;}
                    
        lit_t clit = cls_lit(w, p);
        lit_t len = cls_len(w, p);
        float sign = SIGN(lit^(!pos));

        float orig_loss = loss(snrm2(Z[clit>>1], k), len), new_loss = 0;
        if (!pos) {
            scaddsub(Z[clit>>1], -sign, V[lit>>1], k);
            new_loss = loss(snrm2(Z[clit>>1], k), len-1);
        }
        diff += new_loss-orig_loss;
        if (pos) mix->dual[0][lv-1] -= (len==1)?0.25:(SQRT5/8.);
        else mix->dual[0][lv-1] -= (len==1)?(-0.75):(1/24.);
    }
    return diff;
}

void mix_backtrack(wstack_t *w, mixsat_t *mix, lit_t lit)
{
    // for p in lit^1
    float **V = mix->ov->curr, **Z = mix->oz->curr;
    int k = mix->ov->k;
    for (lit_t **p=w->watched[lit^1]; *p; p++) {
        lit_t clit = cls_lit(w, p);
        float sign = SIGN(lit^1);

        scaddsub(Z[clit>>1], sign, V[lit>>1], k);
    }
}

typedef struct heapnode_t {
    float val;
    void *ptr;
} heapnode_t;

typedef struct heap_t {
    int len, cap;
    heapnode_t *root;
} heap_t;

heap_t heap_init()
{
    int cap = 1024;
    heap_t self = {
        .len = 0, .cap = cap,
        .root = Malloc(cap * sizeof(heapnode_t)),
    };
    return self;
}

void heap_free(heap_t *h)
{
    free(h->root);
}

heapnode_t heap_pop(heap_t *h)
{
    heapnode_t *root = h->root;
    heapnode_t ret = root[0];
    heapnode_t p = root[--h->len];
    int i, c;
    for (i=0; i*2+1 < h->len; i = c) {
        c = i*2+1;
        if (c+1 < h->len && root[c].val < root[c+1].val) c++;
        if (p.val < root[c].val) break;
        else root[i] = root[c];
    }
    root[i] = p;
    return ret;
}

void head_push(heap_t *h, float val, void *ptr)
{
    if (h->cap == h->len) {
        h->cap = ceil(h->cap*1.5);
        h->root = realloc(h->root, h->cap * sizeof *h->root);
        assert(h->root);
    }
    int i = h->len++;
    heapnode_t *root = h->root;
    for(; root[(i-1)/2].val < val; i=(i-1)/2)
        root[i] = root[(i-1)/2];
    root[i] = (heapnode_t){.val = val, .ptr=ptr};
}

// pivot[1:] in ascending certainty, and pivot[0] reserved as sentinal
int bin_maxsat(SATProblem prob, Parameter *param)
{
    int m = prob.m, n = prob.n;
    int lv_thres = 5;

    sol_t best = sol_init(n);
    sol_t curr = sol_init(n);
    wstack_t w = wstack_init(&prob);
    mixsat_t mix = mixsat_init(&prob);

    float *upper = Calloc(n+1, sizeof *upper); // upper bound of sdp
    int   *lower = Calloc(n+1, sizeof *lower); // lower boudn of unsat
    lit_t *pivot = w.pivot, *assign = curr.answer;

    wstack_update(&w, n);

    int lv = n, from_parent = 1;
    upper[lv] = m, best.x = m;
    int state_visited = 0, state_expand = 0, state_prune = 0, diff_cnt = 0;
    int64_t time_st = wall_clock_ns();
    while (lv <= n) {
        if(++state_visited % 10000 == 0) {
            int state_mixed = state_prune+state_expand;
            printf("visited %d mix %d expand %d (%.2f%%) prune %d (%.2f%%) time %.2fs\n",
                    state_visited, state_mixed,
                    state_expand, state_expand*100./state_mixed, 
                    state_prune, state_prune*100./state_mixed, 
                    wall_time_diff(wall_clock_ns(), time_st)
                    );
            fflush(stdout);
        }

        int action;
        if (!from_parent) { // recover and assign next (if exist)
            lit_t lit = assign[lv];
            curr.x -= backtrack(&w, lit);
            mix_backtrack(&w, &mix, lit);

            action = (assign[lv] == pivot[lv] && *w.watched[pivot[lv]^1]) ;
        } else if (lv == 0) { // clearly prunable
            if (curr.x < best.x) save_best(curr, &best, n, time_st);
            action = 0;
        } else if (lv >= lv_thres && upper[lv] >= best.x) { // do mixing
            float fval = 0;
            int mix_val = 0;

            int by_diff = 0;
            if (lv != n && best.x <= ceil(mix.dual[0][lv])){
                mix_val = ceil(mix.dual[0][lv]);
                by_diff = 1;
            }
            if (!by_diff && lv != n && pivot[lv+1] != assign[lv+1]) { // solve the difference problem
                float fval = mixsat_set_var(&mix, &w, lv, 1, assign, curr.x);
                fval += do_mixing(&mix);
                if(fabs(round(fval)-fval) < 1e-4) mix_val = lower[lv]+lround(fval);
                else mix_val = lower[lv] + ceil(fval);
                diff_cnt++;
                if(best.x <= mix_val) by_diff = mix_val-best.x+1;
            }
            
            if (!(best.x <= mix_val)) { // solve the complete problem
                fval = mixsat_set_var(&mix, &w, lv, 0, assign, curr.x);
                fval += do_mixing(&mix);
                fval -= mix.eps;
                mix_val = ceil(curr.x+((fval>0)?fval:0));
            }

            if (best.x <= mix_val) { // prunable
                if(by_diff) state_prune++;
                lower[lv] = mix_val;
                action = 0;
            } else {
                state_expand++;
                int n_trial = param->n_trial*(lv==n?1000*(lv+1):10);
                maxsat_rounding(&mix, &w, n_trial, best.x-curr.x, &curr, &best, time_st);
                mixsat_push(&mix, &w, lv, assign);
                upper[lv] = curr.x+fval;
                mix.dual[0][lv] = curr.x+fval;
                action = 1;
            }
        } else { // from_parent && not prunable
            action = 1;
        }

        if (action) { // try going down
            assign[lv] = pivot[lv] ^ (!*w.watched[pivot[lv]] || !from_parent);

            lit_t lit = assign[lv];
            curr.x += forward(&w, lit);
            float delta = mix_forward(&w, &mix, lit, lv);
            upper[lv-1] = upper[lv] + delta; 
            if(assign[lv]==pivot[lv] || !*w.watched[pivot[lv]]) lower[lv] = m;

            from_parent = 1, lv--;
        } else { // going up
            if (mix.ov->len-1 == lv) mixsat_pop(&mix, &w);
            if (lv != n && lower[lv+1] > lower[lv]) lower[lv+1] = lower[lv];
            from_parent = 0, lv++;
        }
    }
    assert(curr.x==0);
    printf("best %d state_visited %u pruned %u expand %u mix %u time %.2fs\n", best.x, state_visited, state_prune, state_expand, state_prune+state_expand,
                    wall_time_diff(wall_clock_ns(), time_st)
            );
    for(lit_t *p=best.answer+1; p<=best.answer+n; p++)
        printf("%c%u ", (*p&1)?'+':'-', *p>>1);
    printf("\n");
    return best.x;
}

void print_usage(char* prog_name, Parameter *param)
{
    printf( "%s [OPTIONS] INPUT: Mixing method for SDP\n", prog_name); 
    printf( "OPTIONS:\n");
    printf( "\t-s SOLVER: type of solver\n");
    printf( "\t           \"-s maxcut\" for maximum cut\n");
    printf( "\t           \"-s maxsat\" for maximum SAT (default)\n");
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
        .eps = 1e-3,
        .max_iter = 1000,
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
        if(!strcmp(*p, "-s")){
            if(i+1 >= argc) break;
            if(!strcmp(p[1], "maxcut")){
                _param.solver = MAXCUT;
            }else if(!strcmp(p[1], "maxsat")){
                _param.solver = MAXSAT;
            }else {
                int ret = sscanf(p[1], "%d", &_param.solver);
                if(ret != 1 || !(_param.solver >=0 && _param.solver <= 1)) break;
            }
            i++, p++;
        }else if(!strcmp(*p, "-k")){
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
