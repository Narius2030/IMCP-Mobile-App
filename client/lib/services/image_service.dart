import 'dart:async';
import 'dart:developer';
import 'dart:typed_data';

import 'package:client/core/utils/constants.dart';
import 'package:client/models/image_model.dart';
import 'package:dio/dio.dart';
import 'dart:convert';
import 'package:image_picker/image_picker.dart';
import 'dart:ui' as ui;
import 'package:image/image.dart' as img;

class ImageService {
  final dio = Dio();

  Future<Map<String, dynamic>> getImageCaption(
      ImageModel imageModel, String lstmModel, CancelToken cancelToken) async {
    try {
      switch (lstmModel) {
        case "VGG16LM":
          lstmModel = "vgg16lm";
          break;
        case "DarknetVG2":
          lstmModel = "darknetvg2";
          break;
        default:
          lstmModel = "vgg16lm";
          break;
      }
      log(lstmModel);
      var url = "${AppConstants.API_GENERATE_CAPTION}/$lstmModel";
      final response = await dio.post(
        url,
        data: imageModel.toJson(),
        cancelToken: cancelToken,
      );
      log("Response: ${response.data}");
      return response.data;
    } catch (e) {
      throw Exception("Error getting caption for image $e");
    }
  }

  Future<ImageModel> getImageModel(XFile imageFile) async {
    var imagePixel = await getBase64String(imageFile);
    var shape = await getImageShape(imageFile);

    return ImageModel(
      imagePixel: imagePixel,
      shape: shape,
    );
  }

  Future<List<int>> getImageShape(XFile imageFile) async {
    Uint8List imageBytes = await imageFile.readAsBytes();

    final Completer<ui.Image> completer = Completer<ui.Image>();
    ui.decodeImageFromList(imageBytes, (ui.Image img) {
      completer.complete(img);
    });

    ui.Image image = await completer.future;

    int width = image.width;
    int height = image.height;

    int channel = 0;

    img.Image? decodedImage = img.decodeImage(imageBytes);

    if (decodedImage != null) {
      if (decodedImage.hasAlpha) {
        channel = 4; // RGBA
      } else {
        channel = 3; // RGB
      }
    } else {
      log("Failed to decode image");
      channel = 0;
    }

    return [width, height, channel];
  }

  Future<String> getBase64String(XFile imageFile) async {
    final bytes = await imageFile.readAsBytes();
    String base64String = base64Encode(bytes);
    return base64String;
  }

  Future<void> saveUserData(
      ImageModel imageModel, String caption, CancelToken cancelToken) async {
    try {
      var url = "${AppConstants.API_GENERATE_CAPTION}/ingest-user-data";
      await dio.post(
        url,
        data: {
          ...imageModel.toJson(),
          "caption": caption,
        },
        cancelToken: cancelToken,
      );
    } catch (e) {
      throw Exception("Error saving user data $e");
    }
  }

  // String cleanCaption(String caption) {
  //   String cleanedCaption = caption
  //       .replaceAll(RegExp(r'[^a-zA-Z0-9\s.,!?\"]'), '') // Xóa ký tự đặc biệt
  //       .replaceAll(RegExp(r'\s+'), ' ')
  //       .trim();

  //   List<String> words = cleanedCaption.split(' ');

  //   if (words.length > 2) {
  //     words.removeAt(0);
  //     words.removeAt(words.length - 1);
  //   } else {
  //     return '';
  //   }

  //   return words.join(' ');
  // }

  String cleanCaption(String caption) {
    // Use a regular expression to remove content within {}
    String cleanedCaption = caption.replaceAll(RegExp(r'\{.*?\}'), '').trim();

    // Replace multiple spaces with a single space
    cleanedCaption = cleanedCaption.replaceAll(RegExp(r'\s+'), ' ');

    return cleanedCaption;
  }
}
