import 'package:client/presentation/screens/home_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.leanBack);
  runApp(const ImcpApp());
}

class ImcpApp extends StatefulWidget {
  const ImcpApp({super.key});

  @override
  State<ImcpApp> createState() => _ImcpAppState();
}

class _ImcpAppState extends State<ImcpApp> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        appBarTheme: const AppBarTheme(
          systemOverlayStyle: SystemUiOverlayStyle.light,
        ),
      ),
      debugShowCheckedModeBanner: false,
      home: HomePage(),
    );
  }
}
