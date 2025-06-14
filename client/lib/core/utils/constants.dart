import 'package:flutter_dotenv/flutter_dotenv.dart';

class AppConstants {
  static String baseUrl = dotenv.env["API_BASE_URL"]!;

  static String API_GENERATE_CAPTION = "$baseUrl/api/v1";
}
