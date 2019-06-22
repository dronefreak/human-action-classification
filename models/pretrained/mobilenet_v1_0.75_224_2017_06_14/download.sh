#!/bin/bash

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget --continue $( extract_download_url http://www.mediafire.com/file/kibz0x9e7h11ueb/mobilenet_v1_0.75_224.ckpt.data-00000-of-00001 ) -O mobilenet_v1_0.75_224.ckpt.data-00000-of-00001
wget --continue $( extract_download_url http://www.mediafire.com/file/t8909eaikvc6ea2/mobilenet_v1_0.75_224.ckpt.index ) -O mobilenet_v1_0.75_224.ckpt.index
wget --continue $( extract_download_url http://www.mediafire.com/file/6jjnbn1aged614x/mobilenet_v1_0.75_224.ckpt.meta ) -O mobilenet_v1_0.75_224.ckpt.meta
