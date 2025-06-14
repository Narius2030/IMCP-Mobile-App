import 'dart:developer';
import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:client/presentation/widgets/modals/caption_modal_bottom.dart';
import 'package:client/core/utils/colors.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
  }

  void openCaptionBottomSheet(XFile image) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      useRootNavigator: true,
      backgroundColor: const Color(0xFFFAF9F6),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.only(
          topLeft: Radius.circular(20),
          topRight: Radius.circular(20),
        ),
      ),
      builder: (context) => CaptionModalBottom(
        imagePreview: image,
      ),
    );
  }

  Future<void> pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        openCaptionBottomSheet(image);
      }
    } catch (e) {
      log('Error picking image: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CameraAwesomeBuilder.awesome(
        theme: AwesomeTheme(
          bottomActionsBackgroundColor: AppColors.black.withOpacity(0.5),
          buttonTheme: AwesomeButtonTheme(
            backgroundColor: Colors.white.withOpacity(0.5),
            iconSize: 24,
            padding: const EdgeInsets.all(12),
            foregroundColor: Colors.white,
            buttonBuilder: (child, onTap) {
              return ClipOval(
                child: Material(
                  color: Colors.transparent,
                  shape: const CircleBorder(),
                  child: InkWell(
                    splashColor: Colors.white.withOpacity(0.5),
                    highlightColor: Colors.black.withOpacity(0.5),
                    onTap: onTap,
                    child: child,
                  ),
                ),
              );
            },
          ),
        ),
        topActionsBuilder: (state) {
          return Container();
        },
        middleContentBuilder: (state) {
          return Column(
            children: [
              const Spacer(),
              Builder(
                builder: (context) {
                  return Container(
                    margin: const EdgeInsets.only(bottom: 5),
                    child: Align(
                      alignment: Alignment.centerRight,
                      child: RawMaterialButton(
                        onPressed: pickImageFromGallery,
                        elevation: 2.0,
                        fillColor: Colors.transparent.withOpacity(0.5),
                        padding: const EdgeInsets.all(15.0),
                        shape: const CircleBorder(),
                        child: const Icon(
                          CupertinoIcons.photo_on_rectangle,
                          size: 24.0,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  );
                },
              ),
            ],
          );
        },
        onMediaCaptureEvent: (event) {
          switch ((event.status, event.isPicture)) {
            case (MediaCaptureStatus.capturing, true):
              log('Capturing picture...');
            case (MediaCaptureStatus.success, true):
              event.captureRequest.when(
                single: (single) async {
                  XFile image = single.file!;

                  openCaptionBottomSheet(image);
                },
              );
            case (MediaCaptureStatus.failure, true):
              log('Failed to capture picture: ${event.exception}');

            default:
              debugPrint('Unknown event: $event');
          }
        },
        saveConfig: SaveConfig.photo(),
      ),
    );
  }
}
