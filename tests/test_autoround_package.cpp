// Verify every retained tensor and metadata value against the original Q4XL.
#include "../src/gguf.h"
#include <cstdio>
int main(int argc,char** argv){
  try{
    if(argc!=3)return 2;GgufFile source(argv[1]),candidate(argv[2]);size_t kept=0,experts=0;uint64_t bytes=0;
    for(const auto& tensor:source.tensors){const auto& target=candidate.get(tensor.name);
      if(target.dims!=tensor.dims)throw std::runtime_error("shape differs: "+tensor.name);
      if(tensor.name.find("_exps.weight")!=std::string::npos){
        if(target.type!=GGML_AR4_128||target.data)throw std::runtime_error("invalid virtual expert");++experts;
      }else{
        if(target.type!=tensor.type||target.nbytes!=tensor.nbytes||!target.data||memcmp(target.data,tensor.data,tensor.nbytes))
          throw std::runtime_error("retained tensor differs: "+tensor.name);
        ++kept;bytes+=tensor.nbytes;
      }
    }
    for(const auto& key:source.kv_order)if(key.rfind("split.",0)!=0&&key.rfind("inferzero.",0)!=0)
      if(source.kv.at(key)!=candidate.kv.at(key)||source.kv_types.at(key)!=candidate.kv_types.at(key))
        throw std::runtime_error("retained metadata differs: "+key);
    printf("PASS: %zu unchanged tensors (%llu bytes), %zu virtual expert tensors, metadata/tokenizer unchanged\n",kept,(unsigned long long)bytes,experts);
    return kept==1080&&experts==144?0:1;
  }catch(const std::exception& error){fprintf(stderr,"%s\n",error.what());return 1;}
}
