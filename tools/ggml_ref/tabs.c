#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include <stdio.h>
#define D(name,n,fmt) do{FILE*f=fopen("tab_" #name ".bin","wb");fwrite(name,sizeof(name[0]),n,f);fclose(f);}while(0)
int main(){D(iq2xxs_grid,256,0);D(iq2xs_grid,512,0);D(iq2s_grid,1024,0);D(iq1s_grid,2048,0);D(iq3xxs_grid,256,0);D(iq3s_grid,512,0);D(ksigns_iq2xs,128,0);return 0;}
