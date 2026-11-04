// usage: dump TYPEID NELEMS < raw > f32   (ggml's own to_float)
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-quants.h"
int main(int argc, char**argv){
  int t=atoi(argv[1]); long n=atol(argv[2]);
  const struct ggml_type_traits* tr=ggml_get_type_traits(t);
  size_t bs=tr->type_size, be=tr->blck_size; size_t nb=n/be;
  unsigned char*raw=malloc(nb*bs); if(fread(raw,1,nb*bs,stdin)!=nb*bs){fprintf(stderr,"short\n");return 1;}
  float*y=malloc(n*4); if(tr->to_float) tr->to_float(raw,y,n); else if(t==GGML_TYPE_Q8_K) dequantize_row_q8_K((const block_q8_K*)raw,y,n); else {fprintf(stderr,"no to_float\n");return 2;} fwrite(y,4,n,stdout); return 0;}
