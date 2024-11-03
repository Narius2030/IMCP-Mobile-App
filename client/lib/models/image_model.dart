class ImageModel {
  final String imagePixel;
  final List<int> shape;

  ImageModel({required this.imagePixel, required this.shape});

  factory ImageModel.fromJson(Map<String, dynamic> json) {
    return ImageModel(
      imagePixel: json['image_pixel'],
      shape: List<int>.from(json['shape']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'image_pixels': imagePixel,
      'shape': shape,
    };
  }
}
