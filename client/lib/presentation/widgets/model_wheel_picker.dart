import 'package:client/core/utils/colors.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:wheel_chooser/wheel_chooser.dart';

class ModelWheelPicker extends StatelessWidget {
  final List<String> options;
  final Function(dynamic) onValueChanged;

  ModelWheelPicker({required this.options, required this.onValueChanged});

  @override
  Widget build(BuildContext context) {
    return WheelChooser(
      horizontal: true,
      itemSize: 200,
      onValueChanged: onValueChanged,
      datas: options,
      selectTextStyle: const TextStyle(
        color: Color(0xFFD7E0FF),
        fontSize: 16,
        fontWeight: FontWeight.w900,
      ),
      unSelectTextStyle: const TextStyle(
        color: AppColors.grey,
        fontSize: 16,
        fontWeight: FontWeight.w900,
      ),
    );
  }
}
