// usage: quant TYPEID NROWS NCOLS SEED OUTPREFIX : gaussian weights -> ggml_quantize_chunk -> OUT.raw (blocks) OUT.f32 (ggml to_float)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ggml.h"
#include "ggml-quants.h"
static double rn(){double u=(rand()+1.0)/(RAND_MAX+2.0),v=(rand()+1.0)/(RAND_MAX+2.0);return sqrt(-2*log(u))*cos(6.2831853*v);}
int main(int c,char**v){int t=atoi(v[1]);long nr=atol(v[2]),nc=atol(v[3]);srand(atoi(v[4]));
 float*x=malloc(nr*nc*4),*im=malloc(nc*4);for(long i=0;i<nr*nc;i++)x[i]=0.05f*rn()*(1+(i%nc)%7*0.3f);for(long i=0;i<nc;i++)im[i]=0.1f+(float)rand()/RAND_MAX;
 ggml_quantize_init(t);size_t rs=ggml_row_size(t,nc);unsigned char*q=malloc(nr*rs);
 size_t w=ggml_quantize_chunk(t,x,q,0,nr,nc,im);if(w!=nr*rs){fprintf(stderr,"w %zu\n",w);return 1;}
 const struct ggml_type_traits*tr=ggml_get_type_traits(t);float*y=malloc(nr*nc*4);
 if(tr->to_float)tr->to_float(q,y,nr*nc);else dequantize_row_q8_K((const block_q8_K*)q,y,nr*nc);
 char fn[512];snprintf(fn,512,"%s.raw",v[5]);FILE*f=fopen(fn,"wb");fwrite(q,1,nr*rs,f);fclose(f);
 snprintf(fn,512,"%s.f32",v[5]);f=fopen(fn,"wb");fwrite(y,4,nr*nc,f);fclose(f);return 0;}
