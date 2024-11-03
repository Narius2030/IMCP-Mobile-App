import 'dart:developer';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:client/presentation/widgets/modals/caption_modal_bottom.dart';
import 'package:client/core/utils/colors.dart';
import 'package:client/presentation/widgets/segmented_control.dart';
import 'package:client/services/image_service.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';
import 'package:image_picker/image_picker.dart';
import 'package:modal_bottom_sheet/modal_bottom_sheet.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final lstmModel = ["vgg16-lstm", "yolo8-bert-lstm", "yolo8-nobert-ltsm"];
  var chosenModel = "";

  @override
  void initState() {
    super.initState();
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
                        onPressed: () {},
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
              Builder(builder: (context) {
                return Container(
                  height: 40,
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.5),
                  ),
                  child: SegmentedControls(
                    options: lstmModel,
                    onValueChanged: (selected) {
                      chosenModel = selected;
                    },
                  ),
                );
              }),
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
                  if (chosenModel.isEmpty) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text("Please select a model"),
                      ),
                    );
                    return;
                  }

                  XFile image = single.file!;
                  showCupertinoModalBottomSheet(
                    expand: true,
                    context: context,
                    backgroundColor: Colors.transparent,
                    builder: (context) => CaptionModalBottom(
                      imagePreview: image,
                      chosenModel: chosenModel,
                    ),
                  );
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
