#include <metal_stdlib>
#include <c10/metal/expm1f.h>
using namespace metal;

inline float mamba_decay(float z, float h) {
    return abs(z)<0.1f ? fma(c10::metal::expm1f(z),h,h) : precise::exp(z)*h;
}

// STATE_CAPACITY is a bounded power-of-two specialization, not a sequence shape.
kernel void summaries(device const float* x, device const float* dt,
    device const float* A, device const float* B,
    device float* alpha, device float* beta,
    constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint n=id%N, d=(id/N)%D, k=(id/(N*D))%K, b=id/(N*D*K);
    float a=0.0f, h=0.0f;
    for(uint t=k*T; t<min(L,(k+1)*T); ++t) {
        uint i=(b*L+t)*D+d;
        float v=dt[i]*A[d*N+n];
        h=mamba_decay(v,h)+dt[i]*B[(b*L+t)*N+n]*x[i];
        a+=v;
    }
    alpha[id]=a; beta[id]=h;
}

kernel void boundaries(device const float* alpha, device const float* beta,
    device const float* initial, device float* states,
    constant uint& D, constant uint& N, constant uint& K,
    uint id [[thread_position_in_grid]]) {
    uint dn=id%(D*N), b=id/(D*N);
    float h=initial[id];
    states[b*(K+1)*D*N+dn]=h;
    for(uint k=0;k<K;++k) {
        uint i=(b*K+k)*D*N+dn;
        h=mamba_decay(alpha[i],h)+beta[i];
        states[(b*(K+1)+k+1)*D*N+dn]=h;
    }
}

kernel void output_scan(device const float* x, device const float* dt,
    device const float* A, device const float* B, device const float* C,
    device const float* skip, device const float* states, device float* y,
    constant uint& L, constant uint& D, constant uint& N,
    constant uint& K, constant uint& T, uint id [[thread_position_in_grid]]) {
    uint d=id%D, k=(id/D)%K, b=id/(D*K);
    float h[STATE_CAPACITY];
    for(uint n=0;n<N;++n) h[n]=states[((b*(K+1)+k)*D+d)*N+n];
    for(uint t=k*T;t<min(L,(k+1)*T);++t) {
        uint i=(b*L+t)*D+d;
        float value=skip[d]*x[i];
        for(uint n=0;n<N;++n) {
            h[n]=mamba_decay(dt[i]*A[d*N+n],h[n])+dt[i]*B[(b*L+t)*N+n]*x[i];
            value+=h[n]*C[(b*L+t)*N+n];
        }
        y[i]=value;
    }
}
