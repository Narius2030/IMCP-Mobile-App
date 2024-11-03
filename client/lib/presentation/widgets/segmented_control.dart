import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class SegmentedControls extends StatelessWidget {
  final List<String> options;
  final ValueChanged<String> onValueChanged;

  SegmentedControls({required this.options, required this.onValueChanged});

  @override
  Widget build(BuildContext context) {
    return CupertinoSegmentedControl<String>(
      children: {
        for (var option in options) option: Text(option),
      },
      onValueChanged: onValueChanged,
      selectedColor: Colors.white,
      unselectedColor: Colors.black.withOpacity(0.5),
      borderColor: Colors.black,
    );
  }
}
