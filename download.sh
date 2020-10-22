mkdir data
cd data/

wget http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz
tar -xzf EnglishImg.tgz
mv English/ chars74k
rm EnglishImg.tgz

# some of the images for sample 53 have a different format, just remove them
rm chars74k/Img/GoodImg/Bmp/Sample053/img053-00049.png
rm chars74k/Img/GoodImg/Bmp/Sample053/img053-00028.png
rm chars74k/Img/GoodImg/Bmp/Sample053/img053-00024.png
rm chars74k/Img/GoodImg/Bmp/Sample053/img053-00009.png
rm chars74k/Img/GoodImg/Bmp/Sample053/img053-00035.png

cd ..
