#!/bin/bash

. ../utils/parse_yaml.sh
. ../utils/gdownload.sh
. ../utils/conditional.sh

eval $(parse_yaml ../config.yml)
echo 'this is the data_path you are trying to download data into:'
echo $data_path

cd $data_path



# this section is for downloading the Aircraft_fewshot
# md5sum for the downloaded Aircraft_fewshot.tar should be f6646b79d6223af5de175c4849eecf25
echo "downloading Aircraft_fewshot..."
gdownload 1BynNYtQM1i8Rv5_4i52CWhwgSWDwjVkc Aircraft_fewshot.tar
conditional_tar Aircraft_fewshot.tar f6646b79d6223af5de175c4849eecf25

echo ""