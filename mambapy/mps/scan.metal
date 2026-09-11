#include <metal_stdlib>
#include <c10/metal/expm1f.h>
using namespace metal;

inline float mamba_decay(float z, float h) {
    return abs(z)<0.1f ? fma(c10::metal::expm1f(z),h,h) : precise::exp(z)*h;
}

// Retain low-order bits when a slowly decaying state accumulates many updates.
inline void mamba_add(thread float& h, thread float& error, float value) {
    float correction=value+error;
    volatile float total=h+correction;
    error=correction-(total-h);
    h=total;
}
inline void mamba_advance(thread float& h, thread float& error, float z, float u) {
    if(abs(z)<0.1f) {
        float m=c10::metal::expm1f(z);
        mamba_add(h,error,fma(m,h,fma(m,error,u)));
    } else {
        h=fma(precise::exp(z),h+error,u);
        error=0.0f;
    }
}

// STATE_CAPACITY is a bounded power-of-two specialization, not a sequence shape.
kernel void summaries(device const float* x, device const float* dt,
    device const float* A, device const float* B,
    device float* alpha, device float* beta,
    constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint n=id%N, d=(id/N)%D, k=(id/(N*D))%K, b=id/(N*D*K);
    float a=0.0f, h=0.0f, error=0.0f;
    for(uint t=k*T; t<min(L,(k+1)*T); ++t) {
        uint i=(b*L+t)*D+d;
        float v=dt[i]*A[d*N+n];
        mamba_advance(h,error,v,dt[i]*B[(b*L+t)*N+n]*x[i]);
        a+=v;
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

kernel void output_scan(device const float* x, device const float* dt,
    device const float* A, device const float* B, device const float* C,
    device const float* skip, device const float* states, device float* y,
    constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint d=id%D, k=(id/D)%K, b=id/(D*K);
    float h[STATE_CAPACITY], errors[STATE_CAPACITY];
    for(uint n=0;n<N;++n) {
        h[n]=states[((b*(K+1)+k)*D+d)*N+n]; errors[n]=0.0f;
    }
    for(uint t=k*T;t<min(L,(k+1)*T);++t) {
        uint i=(b*L+t)*D+d;
        float value=skip[d]*x[i];
        for(uint n=0;n<N;++n) {
            mamba_advance(h[n],errors[n],dt[i]*A[d*N+n],dt[i]*B[(b*L+t)*N+n]*x[i]);
            value+=h[n]*C[(b*L+t)*N+n];
        }
        y[i]=value;
    }
}

kernel void reverse_summaries(device const float* dt, device const float* A,
    device const float* C, device const float* gy, device float* alpha,
    device float* beta, constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint n=id%N, d=(id/N)%D, k=(id/(N*D))%K, b=id/(N*D*K);
    float zsum=0.0f, q=0.0f, error=0.0f;
    for(int t=int(min(L,(k+1)*T))-1;t>=int(k*T);--t) {
        uint i=(b*L+uint(t))*D+d;
        float z=dt[i]*A[d*N+n];
        mamba_add(q,error,gy[i]*C[(b*L+uint(t))*N+n]);
        mamba_advance(q,error,z,0.0f);
        zsum+=z;
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
kernel void scan_gradients(device const float* x, device const float* dt,
    device const float* A, device const float* B, device const float* C,
    device const float* states, device const float* adjoints, device const float* gy,
    device atomic_float* gx, device atomic_float* gdt, device float* gA,
    device atomic_float* gB, device atomic_float* gC,
    constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint n=id%STATE_CAPACITY, d=(id/STATE_CAPACITY)%D;
    uint k=(id/(STATE_CAPACITY*D))%K, b=id/(STATE_CAPACITY*D*K);
    bool valid=n<N;
    uint count=min(T,L-k*T);
    float hs[257];
    hs[0]=valid ? states[((b*(K+1)+k)*D+d)*N+n] : 0.0f;
    float h=hs[0], error=0.0f;
    for(uint s=0;s<count;++s) {
        uint t=k*T+s, i=(b*L+t)*D+d;
        if(valid) mamba_advance(h,error,dt[i]*A[d*N+n],dt[i]*B[(b*L+t)*N+n]*x[i]);
        hs[s+1]=h;
    }
    float q=valid ? adjoints[((b*(K+1)+k+1)*D+d)*N+n] : 0.0f, ga=0.0f;
    float qerror=0.0f;
    for(int s=int(count)-1;s>=0;--s) {
        uint t=k*T+uint(s), i=(b*L+t)*D+d, j=(b*L+t)*N+n;
        float dx=0.0f, dd=0.0f;
        if(valid) {
            mamba_add(q,qerror,gy[i]*C[j]);
            float ah=mamba_decay(dt[i]*A[d*N+n],hs[s]);
            ga+=q*ah*dt[i];
            dx=q*dt[i]*B[j];
            dd=q*fma(A[d*N+n],ah,B[j]*x[i]);
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
