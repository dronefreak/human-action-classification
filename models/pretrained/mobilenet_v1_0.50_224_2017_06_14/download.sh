#!/bin/bash

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget --continue $( extract_download_url http://www.mediafire.com/file/meu73iq8rxlsd3g/mobilenet_v1_0.50_224.ckpt.data-00000-of-00001 ) -O mobilenet_v1_0.50_224.ckpt.data-00000-of-00001
wget --continue $( extract_download_url http://www.mediafire.com/file/7u6iupfkcaxk5hx/mobilenet_v1_0.50_224.ckpt.index ) -O mobilenet_v1_0.50_224.ckpt.index
wget --continue $( extract_download_url http://www.mediafire.com/file/zp8y4d0ytzharzz/mobilenet_v1_0.50_224.ckpt.meta ) -O mobilenet_v1_0.50_224.ckpt.meta
