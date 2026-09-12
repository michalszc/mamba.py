#include <metal_stdlib>
#include <c10/metal/expm1f.h>
using namespace metal;
using scalar_t = SCALAR_TYPE;

// Degree-six expm1 specialization for |z|<0.1. The omitted term is below
// 2.2e-11, below FP32 rounding here; general arguments still use Metal exp.
inline float small_expm1(float z) {
    return z*fma(z,fma(z,fma(z,fma(z,fma(z,1.0f/720.0f,1.0f/120.0f),
        1.0f/24.0f),1.0f/6.0f),0.5f),1.0f);
}

inline float mamba_decay(float z, float h) {
    if(abs(z)<0.1f) return fma(small_expm1(z),h,h);
    if(z>0.0f && z<1.0f) return fma(c10::metal::expm1f(z),h,h);
    return precise::exp(z)*h;
}

// Retain low-order bits when a slowly decaying state accumulates many updates.
inline void mamba_add(thread float& h, thread float& error, float value) {
    #pragma clang fp reassociate(off)
    float correction=value+error;
    float total=h+correction;
    error=correction-(total-h);
    h=total;
}
inline void mamba_advance(thread float& h, thread float& error, float z, float u) {
    if(abs(z)<0.1f || (z>0.0f && z<1.0f)) {
        float m=abs(z)<0.1f ? small_expm1(z) : c10::metal::expm1f(z);
        mamba_add(h,error,fma(m,h,fma(m,error,u)));
    } else {
        h=fma(precise::exp(z),h+error,u);
        error=0.0f;
    }
}

// STATE_CAPACITY is a bounded power-of-two specialization, not a sequence shape.
kernel void summaries(device const scalar_t* x, device const float* dt,
    device const float* A, device const scalar_t* B,
    device float* alpha, device float* beta,
    constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint n=id%N, d=(id/N)%D, k=(id/(N*D))%K, b=id/(N*D*K);
    float a=0.0f, aerror=0.0f, h=0.0f, error=0.0f;
    for(uint t=k*T; t<min(L,(k+1)*T); ++t) {
        uint i=(b*L+t)*D+d;
        float v=dt[i]*A[d*N+n];
        mamba_advance(h,error,v,dt[i]*B[(b*L+t)*N+n]*x[i]);
        mamba_add(a,aerror,v);
    }
    alpha[id]=a; beta[id]=h;
}

kernel void boundaries(device const float* alpha, device const float* beta,
    device const float* initial, device float* states,
    constant uint& D, constant uint& N, constant uint& K,
    uint id [[thread_position_in_grid]]) {
    uint dn=id%(D*N), b=id/(D*N);
    float h=initial[id], error=0.0f;
    states[b*(K+1)*D*N+dn]=h;
    for(uint k=0;k<K;++k) {
        uint i=(b*K+k)*D*N+dn;
        mamba_advance(h,error,alpha[i],beta[i]);
        states[(b*(K+1)+k+1)*D*N+dn]=h;
    }
}

kernel void output_scan(device const scalar_t* x, device const float* dt,
    device const float* A, device const scalar_t* B, device const scalar_t* C,
    device const float* skip, device float* states, device scalar_t* y,
    device const float* initial, device float* last,
    constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    constexpr uint lanes=STATE_CAPACITY<32 ? STATE_CAPACITY : 32;
    constexpr uint per_lane=(STATE_CAPACITY+lanes-1)/lanes;
    uint lane=id%lanes, d=(id/lanes)%D, k=(id/(lanes*D))%K, b=id/(lanes*D*K);
    float h[per_lane], errors[per_lane], rates[per_lane];
    for(uint s=0;s<per_lane;++s) {
        uint n=lane+s*lanes;
        h[s]=n<N ? (K==1 ? initial[(b*D+d)*N+n] : states[((b*(K+1)+k)*D+d)*N+n]) : 0.0f;
        errors[s]=0.0f;
        rates[s]=n<N ? A[d*N+n] : 0.0f;
        if(K==1 && n<N) states[((b*2)*D+d)*N+n]=h[s];
    }
    for(uint t=k*T;t<min(L,(k+1)*T);++t) {
        uint i=(b*L+t)*D+d;
        float value=0.0f;
        for(uint s=0;s<per_lane;++s) {
            uint n=lane+s*lanes;
            if(n<N) {
                mamba_advance(h[s],errors[s],dt[i]*rates[s],dt[i]*B[(b*L+t)*N+n]*x[i]);
                value+=h[s]*C[(b*L+t)*N+n];
            }
        }
        for(uint offset=1;offset<lanes;offset*=2) value+=simd_shuffle_xor(value,offset);
        if(lane==0) y[i]=scalar_t(value+skip[d]*x[i]);
    }
    for(uint s=0;s<per_lane;++s) {
        uint n=lane+s*lanes;
        if(n<N) {
            if(K==1) states[((b*2+1)*D+d)*N+n]=h[s];
            if(k==K-1) last[(b*D+d)*N+n]=h[s];
        }
    }
}

kernel void reverse_summaries(device const float* dt, device const float* A,
    device const scalar_t* C, device const float* gy, device float* alpha,
    device float* beta, constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint n=id%N, d=(id/N)%D, k=(id/(N*D))%K, b=id/(N*D*K);
    float zsum=0.0f, zerror=0.0f, q=0.0f, error=0.0f;
    for(int t=int(min(L,(k+1)*T))-1;t>=int(k*T);--t) {
        uint i=(b*L+uint(t))*D+d;
        float z=dt[i]*A[d*N+n];
        mamba_add(q,error,gy[i]*C[(b*L+uint(t))*N+n]);
        mamba_advance(q,error,z,0.0f);
        mamba_add(zsum,zerror,z);
    }
    alpha[id]=zsum; beta[id]=q;
}

kernel void reverse_boundaries(device const float* alpha, device const float* beta,
    device const float* final_grad, device float* adjoints,
    constant uint& D, constant uint& N, constant uint& K,
    uint id [[thread_position_in_grid]]) {
    uint dn=id%(D*N), b=id/(D*N);
    float q=final_grad[id], error=0.0f;
    adjoints[(b*(K+1)+K)*D*N+dn]=q;
    for(int k=int(K)-1;k>=0;--k) {
        uint i=(b*K+uint(k))*D*N+dn;
        mamba_advance(q,error,alpha[i],beta[i]);
        adjoints[(b*(K+1)+uint(k))*D*N+dn]=q;
    }
}

// Each thread recomputes one state coordinate for at most 256 timesteps.
// Reductions accumulate directly into compact input-shaped gradient buffers.
kernel void scan_gradients(device const scalar_t* x, device const float* dt,
    device const float* A, device const scalar_t* B, device const scalar_t* C,
    device const float* states, device const float* adjoints, device const float* gy,
    device atomic_float* gx, device atomic_float* gdt, device float* gA,
    device atomic_float* gB, device atomic_float* gC,
    constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint n=id%STATE_CAPACITY, d=(id/STATE_CAPACITY)%D;
    uint k=(id/(STATE_CAPACITY*D))%K, b=id/(STATE_CAPACITY*D*K);
    bool valid=n<N;
    uint count=min(T,L-k*T);
    float hs[CHUNK_CAPACITY+1];
    hs[0]=valid ? states[((b*(K+1)+k)*D+d)*N+n] : 0.0f;
    float h=hs[0], error=0.0f;
    for(uint s=0;s<count;++s) {
        uint t=k*T+s, i=(b*L+t)*D+d;
        if(valid) mamba_advance(h,error,dt[i]*A[d*N+n],dt[i]*B[(b*L+t)*N+n]*x[i]);
        hs[s+1]=h;
    }
    float q=valid ? adjoints[((b*(K+1)+k+1)*D+d)*N+n] : 0.0f, ga=0.0f;
    float qerror=0.0f, gaerror=0.0f;
    for(int s=int(count)-1;s>=0;--s) {
        uint t=k*T+uint(s), i=(b*L+t)*D+d, j=(b*L+t)*N+n;
        float dx=0.0f, dd=0.0f;
        if(valid) {
            mamba_add(q,qerror,gy[i]*C[j]);
            float ah=mamba_decay(dt[i]*A[d*N+n],hs[s]);
            mamba_add(ga,gaerror,q*ah*dt[i]);
            dx=q*dt[i]*B[j];
            dd=q*fma(A[d*N+n],ah,float(B[j])*float(x[i]));
            atomic_fetch_add_explicit(gB+j,q*dt[i]*x[i],memory_order_relaxed);
            atomic_fetch_add_explicit(gC+j,gy[i]*hs[s+1],memory_order_relaxed);
            mamba_advance(q,qerror,dt[i]*A[d*N+n],0.0f);
        }
        for(uint offset=1;offset<min(uint(STATE_CAPACITY),32u);offset*=2) {
            dx+=simd_shuffle_xor(dx,offset);
            dd+=simd_shuffle_xor(dd,offset);
        }
        if(n%min(uint(STATE_CAPACITY),32u)==0) {
            atomic_fetch_add_explicit(gx+i,dx,memory_order_relaxed);
            atomic_fetch_add_explicit(gdt+i,dd,memory_order_relaxed);
        }
    }
    if(valid) gA[((b*K+k)*D+d)*N+n]=ga;
}

kernel void reduce_A(device const float* partial, device float* result,
    constant uint& DN, constant uint& BK, uint id [[thread_position_in_grid]]) {
    float sum=0.0f, error=0.0f;
    for(uint i=0;i<BK;++i) mamba_add(sum,error,partial[i*DN+id]);
    result[id]=sum;
}

// Two FP32 components retain the low part of cancelling long-sequence sums.
inline float2 pair_add(float2 a, float2 b) {
    #pragma clang fp reassociate(off)
    float s=a.x+b.x, v=s-a.x;
    float e=(a.x-(s-v))+(b.x-v)+a.y+b.y;
    float h=s+e;
    return float2(h,e-(h-s));
}
inline float2 pair_mul(float2 a, float2 b) {
    float p=a.x*b.x;
    float e=fma(a.x,b.x,-p)+a.x*b.y+a.y*b.x;
    return pair_add(float2(p,0.0f),float2(e,0.0f));
}
inline float2 pair_exp(float2 z) {
    if(abs(z.x)>=0.1f) return float2(precise::exp(z.x),0.0f);
    // Taylor through degree eight; remainder <3e-15 over this interval.
    float2 term=float2(1.0f,0.0f), sum=term;
    for(uint i=1;i<=8;++i) {
        term=pair_mul(term,z);
        float q=term.x/float(i);
        term=pair_add(float2(q,0.0f),float2((fma(-q,float(i),term.x)+term.y)/float(i),0.0f));
        sum=pair_add(sum,term);
    }
    return sum;
}

// Forward sensitivities avoid compounding forward/reverse boundary errors.
// This accuracy specialization is limited to long sequences; normal training
// uses the parallel reverse scan above.
kernel void long_A_gradient(device const scalar_t* x, device const float* dt,
    device const float* A, device const scalar_t* B, device const scalar_t* C,
    device const float* states, device const float* gy, device const float* gh,
    device float* result, constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& batches, uint id [[thread_position_in_grid]]) {
    uint n=id%N, d=id/N;
    float2 sum=float2(0.0f);
    for(uint b=0;b<batches;++b) {
        float2 h=float2(states[((b*(K+1))*D+d)*N+n],0.0f), sensitivity=float2(0.0f);
        for(uint t=0;t<L;++t) {
            uint i=(b*L+t)*D+d, j=(b*L+t)*N+n;
            float2 delta=float2(dt[i],0.0f);
            float2 decay=pair_exp(pair_mul(delta,float2(A[id],0.0f)));
            float2 ah=pair_mul(decay,h);
            sensitivity=pair_add(pair_mul(decay,sensitivity),pair_mul(delta,ah));
            h=pair_add(ah,pair_mul(pair_mul(delta,float2(B[j],0.0f)),float2(x[i],0.0f)));
            sum=pair_add(sum,pair_mul(pair_mul(float2(gy[i],0.0f),float2(C[j],0.0f)),sensitivity));
        }
        sum=pair_add(sum,pair_mul(float2(gh[b*D*N+id],0.0f),sensitivity));
    }
    result[id]=sum.x;
}
