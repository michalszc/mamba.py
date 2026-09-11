#include <metal_stdlib>
using namespace metal;
using scalar_t = SCALAR_TYPE;

kernel void conv_forward(device const scalar_t* x, device const scalar_t* w,
    device const scalar_t* bias, device scalar_t* y,
    constant uint& L, constant uint& D, constant uint& W,
    uint id [[thread_position_in_grid]]) {
    uint d=id%D,t=(id/D)%L,b=id/(D*L),S=L+W-1;
    float v=bias[d];
    for(uint j=0;j<W;++j) v+=float(x[(b*S+t+j)*D+d])*float(w[d*W+j]);
    y[id]=scalar_t(v/(1.0f+exp(-v)));
}

kernel void conv_activation_grad(device const scalar_t* x, device const scalar_t* w,
    device const scalar_t* bias, device const float* gy, device float* gv,
    constant uint& L, constant uint& D, constant uint& W,
    uint id [[thread_position_in_grid]]) {
    uint d=id%D,t=(id/D)%L,b=id/(D*L),S=L+W-1;
    float v=bias[d];
    for(uint j=0;j<W;++j) v+=float(x[(b*S+t+j)*D+d])*float(w[d*W+j]);
    float s=1.0f/(1.0f+exp(-v));
    gv[id]=gy[id]*s*(1.0f+v*(1.0f-s));
}

kernel void conv_input_grad(device const float* gv, device const scalar_t* w,
    device float* gx, constant uint& L, constant uint& D, constant uint& W,
    uint id [[thread_position_in_grid]]) {
    uint S=L+W-1,d=id%D,t=(id/D)%S,b=id/(D*S);
    float v=0.0f;
    for(uint j=0;j<W;++j) {
        int pos=int(t)-int(j);
        if(pos>=0 && pos<int(L)) v+=gv[(b*L+uint(pos))*D+d]*w[d*W+j];
    }
    gx[id]=v;
}

kernel void conv_weight_grad(device const scalar_t* x, device const float* gv,
    device float* gw, constant uint& B, constant uint& L,
    constant uint& D, constant uint& W, uint id [[thread_position_in_grid]]) {
    uint d=id/W,j=id%W,S=L+W-1;
    float v=0.0f,error=0.0f;
    for(uint b=0;b<B;++b) for(uint t=0;t<L;++t) {
        float value=gv[(b*L+t)*D+d]*x[(b*S+t+j)*D+d]+error;
        volatile float next=v+value;
        error=value-(next-v); v=next;
    }
    gw[id]=v;
}
