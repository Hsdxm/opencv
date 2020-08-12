opencv version：4.4.0

install FFmpeg:
	git clone https://github.com/FFmpeg/FFmpeg.git
	cd FFmpeg
	./configure --enable-shared
	make -j8
	make install

install opencv:
	export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
	mkdir build
	cd build
	cmake .. -DBUILD_opencv_world=ON
	make -j8
	make install


part1:
	test dnn module: darknet yolov3
