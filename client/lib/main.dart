import 'package:client/core/utils/injection_container.dart';
import 'package:client/presentation/screens/home_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:google_fonts/google_fonts.dart';

void main() async {
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.leanBack);
  configureDependencies();
  await setup();
  runApp(const ImcpApp());
}

Future<void> setup() async {
  await dotenv.load(fileName: ".env");
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
      theme: _buildTheme(Brightness.light),
      debugShowCheckedModeBanner: false,
      home: HomePage(),
    );
  }

  ThemeData _buildTheme(brightness) {
    var baseTheme = ThemeData(
      brightness: brightness,
      appBarTheme: const AppBarTheme(
        systemOverlayStyle: SystemUiOverlayStyle.light,
      ),
    );

    return baseTheme.copyWith(
      textTheme: GoogleFonts.quicksandTextTheme(baseTheme.textTheme).copyWith(
        bodyMedium: GoogleFonts.quicksand(
          textStyle: const TextStyle(fontWeight: FontWeight.w600),
        ),
      ),
    );
  }
}
