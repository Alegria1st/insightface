
: <<'END'
The final directory structure is as follows.
WebFace datasets(train.idx, train.lst, train.rex) should be moved to the corresponding path in advance.

dataset --> main directory(Folder to designate as docker volume)
├── Dockers-main --> Docker build file
├── WebFace42M --> WebFace Full data (folder num 0 ~ 9)
├── WebFace4M --> WebFace Sample (folder num  0)
├── WebFace12M --> WebFace Sample (folder num  0, 1, 2)
├── validation --> ijb, lfw.bin, cfp_fp.bin, agedb_30.bin
├── arcface --> insightface github 
END

# download docker images for insightface
git clone https://github.com/Alegria1st/Dockers.git

# build docker images
docker build --tag nist:1.1 ./Dockers-main2

# make container images 
docker run --name bentley1 -d -i -t --gpus all --ipc=host  -v /home/kakao-ai-devteam/dataset:/app nist:1.0
docker start bentley1
docker attach bentley1

# clone 
git clone https://github.com/Alegria1st/insightface.git 

# move file to recognition/arcface_torch/train_tmp
mkdir -p recognition/arcface_torch/train_tmp
mv WebFace42M recognition/arcface_torch/train_tmp
mv WebFace4M recognition/arcface_torch/train_tmp
## mv WebFace12M recognition/arcface_torch/train_tmp
