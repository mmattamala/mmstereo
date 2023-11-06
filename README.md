# mmstereo

Training code from [mmstereo](https://github.com/kevinleestone/mmstereo) with ROS wrappers.

## Docker

To build the image:

```sh
docker buildx build --no-cache -t mmattamala/mmstereo:pytorch2.0 -f Dockerfile .
```

To run the image:

```sh
docker run -it --runtime nvidia -v "$PWD"/mmstereo:/workspace/mmstereo mmattamala/mmstereo:pytorch2.0
```
