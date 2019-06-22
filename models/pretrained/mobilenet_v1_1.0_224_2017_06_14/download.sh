#!/bin/bash

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget --continue $( extract_download_url http://www.mediafire.com/file/oh6njnz9lgoqwdj/mobilenet_v1_1.0_224.ckpt.data-00000-of-00001 ) -O mobilenet_v1_1.0_224.ckpt.data-00000-of-00001
wget --continue $( extract_download_url http://www.mediafire.com/file/61qln0tbac4ny9o/mobilenet_v1_1.0_224.ckpt.meta ) -O mobilenet_v1_1.0_224.ckpt.meta
wget --continue $( extract_download_url http://www.mediafire.com/file/2111rh6tb5fl1lr/mobilenet_v1_1.0_224.ckpt.index ) -O mobilenet_v1_1.0_224.ckpt.index
