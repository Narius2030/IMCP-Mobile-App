import 'dart:developer';
import 'dart:io';

import 'package:client/core/utils/colors.dart';
import 'package:client/models/image_model.dart';
import 'package:client/presentation/screens/photo_view.dart';
import 'package:client/presentation/widgets/shimmers/caption_shimmer.dart';
import 'package:client/services/image_service.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart'; // Import the package
import 'package:get_it/get_it.dart';
import 'package:image_picker/image_picker.dart';
import 'package:photo_view/photo_view.dart';
import 'package:typewritertext/typewritertext.dart';
import 'dart:math' as math;

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
  final CancelToken cancelToken = CancelToken();
  final FlutterTts flutterTts = FlutterTts();

  void unawaited(Future<void> future) {
    future.catchError((error, stackTrace) {
      log("Uncaught Future error: $error", stackTrace: stackTrace);
    });
  }

  Future<String> getImageCaption() async {
    try {
      var imageModel = await imageService.getImageModel(widget.imagePreview);
      var response = await imageService.getImageCaption(
        imageModel,
        widget.chosenModel,
        cancelToken,
      );
      String caption = response["predicted_caption"];

      unawaited(saveUserDataInBackground(imageModel, caption));

      String cleanCaption = imageService.cleanCaption(caption);
      await speakCaption(cleanCaption);
      return cleanCaption;
    } catch (e) {
      log("Error getting caption for image $e");
      throw Exception("Error getting caption for image $e");
    }
  }

  Future<void> saveUserDataInBackground(
      ImageModel imageModel, String caption) async {
    try {
      // Chạy API lưu dữ liệu trong microtask
      await Future.microtask(
        () => imageService.saveUserData(
          imageModel,
          caption,
          cancelToken,
        ),
      );
      log("User data saved successfully");
    } catch (e) {
      log("Error saving user data: $e");
    }
  }

  Future<void> speakCaption(String caption) async {
    await flutterTts.setLanguage("en-US");
    await flutterTts.setPitch(1.0);
    await flutterTts.setSpeechRate(0.5);
    await flutterTts.speak(caption);
  }

  @override
  void initState() {
    super.initState();
    _future = getImageCaption();
  }

  @override
  void dispose() {
    cancelToken.cancel();
    flutterTts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.7,
      minChildSize: 0.5,
      maxChildSize: 0.9,
      expand: false,
      builder: (context, scrollController) {
        return Padding(
          padding: const EdgeInsets.all(12.0),
          child: SingleChildScrollView(
            controller: scrollController,
            child: DecoratedBox(
              decoration: const BoxDecoration(
                color: Color(0xFFFAF9F6),
              ),
              child: Column(
                children: [
                  Align(
                    alignment: Alignment.topRight,
                    child: IconButton(
                      color: const Color(0xFF212121),
                      onPressed: () {
                        Navigator.pop(context);
                      },
                      icon: const Icon(
                        Icons.close_rounded,
                        size: 30,
                      ),
                    ),
                  ),
                  Stack(
                    children: [
                      InkWell(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => ViewPhotoPage(
                                imagePreview: widget.imagePreview,
                              ),
                            ),
                          );
                        },
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(10),
                          child: Image.file(
                            File(widget.imagePreview.path),
                            width: double.infinity,
                            height: 200,
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.only(
                          top: 4.0,
                          right: 4.0,
                        ),
                        child: Align(
                          alignment: Alignment.bottomRight,
                          child: Transform(
                              alignment: Alignment.center,
                              transform: Matrix4.rotationY(math.pi),
                              child: RawMaterialButton(
                                splashColor: Colors.transparent,
                                onPressed: () {},
                                elevation: 2.0,
                                fillColor: Colors.black.withOpacity(0.2),
                                constraints:
                                    const BoxConstraints(minWidth: 0.0),
                                padding: const EdgeInsets.all(10.0),
                                shape: const CircleBorder(),
                                child: const Icon(
                                  Icons.search_outlined,
                                  size: 20.0,
                                  color: Colors.white,
                                ),
                              )),
                        ),
                      ),
                    ],
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
                          textAlign: TextAlign.start,
                          maxLines: 100,
                          maintainSize: false,
                          duration: const Duration(milliseconds: 10),
                          snapshot.data.toString(),
                          style: const TextStyle(
                            shadows: [
                              Shadow(
                                color: Colors.black12,
                                offset: Offset(0, 1),
                                blurRadius: 10,
                              ),
                            ],
                            fontSize: 20,
                            color: Color(0xFF212121),
                            letterSpacing: 0.5,
                          ),
                        );
                      }
                    },
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
