FROM python:3.7-alpine3.10
WORKDIR /root
ADD . /root/class-tracker/

RUN mkdir -p /root/class-tracker/build && \
	echo "http://nl.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories && \
	apk add --update --no-cache g++ make eigen-dev py3-numpy-dev cmake && \
	python3 -m pip install pytest && \
	ln -s /usr/lib/python3.8/site-packages/numpy/core/include/numpy/ /usr/local/include/numpy && \
	wget https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz && \
	tar xf v2.5.0.tar.gz && \
	rm -f *.gz && \
	cd /root/pybind* && \
	mkdir build && \
	cd build && \
	cmake -DDOWNLOAD_CATCH=ON .. && \
	make && \
	make install && \
	cd /root/class-tracker/build && \
	sed -i 's/Eigen\//eigen3\/Eigen\//g' /usr/local/include/pybind11/eigen.h && \
	cmake .. && \
	make && \
	cp track.*.so /root/ && \
	cd /root && \
	rm -R -- */