import 'package:client/services/image_service.dart';
import 'package:get_it/get_it.dart';

final locator = GetIt.instance;

//Cai nay de dang ky dependency
void configureDependencies() {
  locator.registerSingleton<ImageService>(ImageService());
}
