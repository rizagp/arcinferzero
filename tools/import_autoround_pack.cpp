// Write a hybrid package manifest: calibrated expert records are imported separately.
// Preserve all non-expert source tensors, metadata and tokenizer byte for byte.
// usage: import_autoround_pack SOURCE.gguf OUTPUT.gguf EXPERTS_BASENAME SOURCE_REVISION
#include "../zero/gguf_pack.h"
int main(int argc, char** argv) {
  try {
    if(argc!=5)throw std::runtime_error("usage: import_autoround_pack SOURCE.gguf OUTPUT.gguf EXPERTS_BASENAME REVISION");
    namespace fs=std::filesystem;
    if(fs::exists(argv[2]))throw std::runtime_error("refusing to overwrite an existing model");
    const std::string expert_name=argv[3];
    if(expert_name.empty()||expert_name.find_first_of("/\\:")!=std::string::npos)
      throw std::runtime_error("expert file must be a basename");
    GgufFile src(argv[1]);
    if(src.get_str("general.architecture")!="qwen4exp"||src.get_int("qwen4exp.block_count")!=48||
       src.get_int("qwen4exp.expert_count")!=512||src.get_int("qwen4exp.embedding_length")!=2560)
      throw std::runtime_error("expected Flash Next 48-layer/512-expert source");
    std::vector<const GgufTensor*> keep;
    std::vector<std::string> names;std::vector<int64_t> types,dims;
    for(const auto& tensor:src.tensors) {
      if(zpack::is_expert_tensor(tensor.name)) {
        if(tensor.dims.size()!=3||tensor.dims[2]!=512)throw std::runtime_error("unexpected expert shape");
        names.push_back(tensor.name);types.push_back(GGML_AR4_128);
        for(int d=0;d<4;++d)dims.push_back(d<(int)tensor.dims.size()?tensor.dims[d]:0);
      } else {if(!tensor.data)throw std::runtime_error("source has virtual non-expert data");keep.push_back(&tensor);}
    }
    if(names.size()!=144)throw std::runtime_error("expected 144 target expert tensors");
    std::vector<std::string> keys;
    for(const auto& key:src.kv_order)if(key.rfind("split.",0)!=0&&key.rfind("inferzero.",0)!=0)keys.push_back(key);
    const std::string temporary=std::string(argv[2])+".tmp";
    zpack::Writer out(temporary);out.put<uint32_t>(0x46554747);out.put<uint32_t>(3);
    out.put<uint64_t>(keep.size());out.put<uint64_t>(keys.size()+8);
    for(const auto& key:keys){const auto ty=src.kv_types.at(key);zpack::put_kv(out,key,ty.first,ty.second,src.kv.at(key));}
    zpack::put_kv(out,"inferzero.package",4,0,GgufValue((int64_t)1));
    zpack::put_kv(out,"inferzero.experts_file",8,0,GgufValue(expert_name));
    zpack::put_kv(out,"inferzero.source",8,0,GgufValue(fs::absolute(argv[1]).string()));
    zpack::put_kv(out,"inferzero.virtual.names",9,8,GgufValue(names));
    zpack::put_kv(out,"inferzero.virtual.types",9,4,GgufValue(types));
    zpack::put_kv(out,"inferzero.virtual.dims",9,11,GgufValue(dims));
    zpack::put_kv(out,"inferzero.autoround.model",8,0,GgufValue("Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"));
    zpack::put_kv(out,"inferzero.autoround.revision",8,0,GgufValue(std::string(argv[4])));
    const uint64_t align=src.get_int("general.alignment",32);uint64_t offset=0;
    for(const auto* t:keep){out.str(t->name);out.put<uint32_t>(t->dims.size());for(auto d:t->dims)out.put<uint64_t>(d);
      out.put<uint32_t>(t->type);out.put<uint64_t>(offset);offset+=(t->nbytes+align-1)/align*align;}
    out.pad(align);
    for(const auto* t:keep){out.put(t->data,t->nbytes);out.pad(align);}
    fprintf(stderr,"hybrid package: %zu unchanged tensors, %zu virtual expert tensors, %.3f GiB\n",keep.size(),names.size(),out.pos()/1073741824.0);
    out.close();fs::rename(temporary,argv[2]);return 0;
  }catch(const std::exception& error){fprintf(stderr,"import_autoround_pack: %s\n",error.what());return 1;}
}
