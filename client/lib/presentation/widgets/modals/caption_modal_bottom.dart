import 'dart:developer';
import 'dart:io';

import 'package:client/presentation/widgets/shimmers/caption_shimmer.dart';
import 'package:client/services/image_service.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';
import 'package:image_picker/image_picker.dart';
import 'package:typewritertext/typewritertext.dart';

class CaptionModalBottom extends StatefulWidget {
  final XFile imagePreview;
  final String chosenModel;

  const CaptionModalBottom({
    super.key,
    required this.imagePreview,
    required this.chosenModel,
  });

  @override
  State<CaptionModalBottom> createState() => _CaptionModalBottomState();
}

class _CaptionModalBottomState extends State<CaptionModalBottom> {
  var imageService = GetIt.I<ImageService>();
  late Future _future;
  CancelToken cancelToken = CancelToken();

  Future<String> getImageCaption() async {
    try {
      var imageModel = await imageService.getImageModel(widget.imagePreview);
      var response = await imageService.getImageCaption(
        imageModel,
        widget.chosenModel,
        cancelToken,
      );
      return response["predicted_caption"];
    } catch (e) {
      log("Error getting caption for image $e");
      throw Exception("Error getting caption for image $e");
    }
  }

  @override
  void initState() {
    super.initState();
    _future = getImageCaption();
  }

  @override
  void dispose() {
    cancelToken.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Material(
        child: SafeArea(
      top: false,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Align(
                  alignment: Alignment.topRight,
                  child: IconButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    icon: Icon(Icons.close),
                  ),
                ),
                ClipRRect(
                  borderRadius: BorderRadius.circular(5),
                  child: Image.file(
                    File(widget.imagePreview.path),
                    width: double.infinity,
                    height: 200,
                    fit: BoxFit.cover,
                  ),
                ),
                const SizedBox(height: 10),
                FutureBuilder(
                  future: _future,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const CaptionShimmer();
                    } else if (snapshot.hasError) {
                      return Text("Error: ${snapshot.error}");
                    } else {
                      return TypeWriter.text(
                        duration: const Duration(milliseconds: 20),
                        snapshot.data.toString(),
                        style: const TextStyle( 
                          shadows: [
                            Shadow(
                              color: Colors.black12,
                              offset: Offset(0, 1),
                              blurRadius: 2,
                            ),
                          ],
                          fontSize: 20,
                          fontWeight: FontWeight.w500,
                          color: Colors.black87,
                          fontFamily: 'Roboto',
                          letterSpacing: 0.5,
                        ),
                      );
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    ));
  }
}
