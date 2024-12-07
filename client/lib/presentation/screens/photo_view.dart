import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:photo_view/photo_view.dart';

class ViewPhotoPage extends StatefulWidget {
  final XFile imagePreview;
  const ViewPhotoPage({
    super.key,
    required this.imagePreview,
  });

  @override
  State<ViewPhotoPage> createState() => _ViewPhotoPageState();
}

class _ViewPhotoPageState extends State<ViewPhotoPage> {
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.black,
          foregroundColor: Colors.white,
          leading: IconButton(
            onPressed: () {
              Navigator.pop(context);
            },
            icon: const Icon(
              Icons.chevron_left,
              size: 30,
            ),
          ),
        ),
        backgroundColor: Colors.black,
        body: PhotoView(
          imageProvider: FileImage(File(widget.imagePreview.path)),
        ),
      ),
    );
  }
}
