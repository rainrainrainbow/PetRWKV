import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

/// RWKV 推理引擎封装
/// 基于 rwkv-mobile 的 C++ 推理引擎，通过 FFI 调用
class RWKVInferenceEngine {
  static final RWKVInferenceEngine _instance = RWKVInferenceEngine._internal();
  factory RWKVInferenceEngine() => _instance;
  RWKVInferenceEngine._internal();
  
  late DynamicLibrary _lib;
  late RWKVBindings _bindings;
  bool _initialized = false;
  
  /// 初始化引擎
  Future<void> initialize() async {
    if (_initialized) return;
    
    try {
      if (Platform.isAndroid) {
        _lib = DynamicLibrary.open('librwkv_engine.so');
      } else if (Platform.isIOS) {
        _lib = DynamicLibrary.process();
      } else if (Platform.isWindows) {
        _lib = DynamicLibrary.open('rwkv_engine.dll');
      } else if (Platform.isLinux) {
        _lib = DynamicLibrary.open('librwkv_engine.so');
      } else if (Platform.isMacOS) {
        _lib = DynamicLibrary.open('librwkv_engine.dylib');
      }
      
      _bindings = RWKVBindings(_lib);
      _initialized = true;
    } catch (e) {
      throw Exception('Failed to initialize RWKV engine: $e');
    }
  }
  
  /// 加载模型
  Future<void> loadModel(String modelPath, {String? backend}) async {
    if (!_initialized) {
      throw Exception('Engine not initialized');
    }
    
    final pathPtr = modelPath.toNativeUtf8();
    final backendPtr = backend?.toNativeUtf8();
    
    try {
      final result = _bindings.load_model(pathPtr, backendPtr);
      if (result != 0) {
        throw Exception('Failed to load model: error code $result');
      }
    } finally {
      calloc.free(pathPtr);
      if (backendPtr != null) calloc.free(backendPtr);
    }
  }
  
  /// 文本生成
  Future<String> generate({
    required String prompt,
    int maxTokens = 256,
    double temperature = 0.8,
    double topP = 0.9,
  }) async {
    if (!_initialized) {
      throw Exception('Engine not initialized');
    }
    
    final promptPtr = prompt.toNativeUtf8();
    
    try {
      final resultPtr = _bindings.generate(
        promptPtr,
        maxTokens,
        temperature,
        topP,
      );
      
      if (resultPtr == nullptr) {
        throw Exception('Generation failed');
      }
      
      final result = resultPtr.toDartString();
      calloc.free(resultPtr);
      
      return result;
    } finally {
      calloc.free(promptPtr);
    }
  }
  
  /// 多模态理解（图片）
  Future<String> understandImage({
    required String imagePath,
    required String prompt,
    int maxTokens = 256,
  }) async {
    if (!_initialized) {
      throw Exception('Engine not initialized');
    }
    
    final imagePtr = imagePath.toNativeUtf8();
    final promptPtr = prompt.toNativeUtf8();
    
    try {
      final resultPtr = _bindings.understand_image(
        imagePtr,
        promptPtr,
        maxTokens,
      );
      
      if (resultPtr == nullptr) {
        throw Exception('Image understanding failed');
      }
      
      final result = resultPtr.toDartString();
      calloc.free(resultPtr);
      
      return result;
    } finally {
      calloc.free(imagePtr);
      calloc.free(promptPtr);
    }
  }
  
  /// 多模态理解（音频）
  Future<String> understandAudio({
    required String audioPath,
    required String prompt,
    int maxTokens = 256,
  }) async {
    if (!_initialized) {
      throw Exception('Engine not initialized');
    }
    
    final audioPtr = audioPath.toNativeUtf8();
    final promptPtr = prompt.toNativeUtf8();
    
    try {
      final resultPtr = _bindings.understand_audio(
        audioPtr,
        promptPtr,
        maxTokens,
      );
      
      if (resultPtr == nullptr) {
        throw Exception('Audio understanding failed');
      }
      
      final result = resultPtr.toDartString();
      calloc.free(resultPtr);
      
      return result;
    } finally {
      calloc.free(audioPtr);
      calloc.free(promptPtr);
    }
  }
  
  /// 获取文本嵌入（用于 RAG）
  Future<List<double>> getEmbedding(String text) async {
    if (!_initialized) {
      throw Exception('Engine not initialized');
    }
    
    final textPtr = text.toNativeUtf8();
    
    try {
      final resultPtr = _bindings.get_embedding(textPtr);
      
      if (resultPtr == nullptr) {
        throw Exception('Embedding failed');
      }
      
      // 解析嵌入向量
      final embeddingStr = resultPtr.toDartString();
      calloc.free(resultPtr);
      
      final values = embeddingStr.split(',').map((s) => double.parse(s)).toList();
      return values;
    } finally {
      calloc.free(textPtr);
    }
  }
  
  /// 释放资源
  void dispose() {
    if (_initialized) {
      _bindings.unload_model();
      _initialized = false;
    }
  }
}

/// C 函数绑定定义
class RWKVBindings {
  final DynamicLibrary _lib;
  
  RWKVBindings(this._lib);
  
  // int load_model(const char* model_path, const char* backend)
  late final int Function(Pointer<Utf8>, Pointer<Utf8>) load_model = 
      _lib.lookup<NativeFunction<Int32 Function(Pointer<Utf8>, Pointer<Utf8>)>>('load_model').asFunction();
  
  // void unload_model()
  late final void Function() unload_model = 
      _lib.lookup<NativeFunction<Void Function()>>('unload_model').asFunction();
  
  // char* generate(const char* prompt, int max_tokens, float temperature, float top_p)
  late final Pointer<Utf8> Function(Pointer<Utf8>, Int32, Float, Float) generate = 
      _lib.lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Int32, Float, Float)>>('generate').asFunction();
  
  // char* understand_image(const char* image_path, const char* prompt, int max_tokens)
  late final Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>, Int32) understand_image = 
      _lib.lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>, Int32)>>('understand_image').asFunction();
  
  // char* understand_audio(const char* audio_path, const char* prompt, int max_tokens)
  late final Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>, Int32) understand_audio = 
      _lib.lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>, Pointer<Utf8>, Int32)>>('understand_audio').asFunction();
  
  // char* get_embedding(const char* text)
  late final Pointer<Utf8> Function(Pointer<Utf8>) get_embedding = 
      _lib.lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Utf8>)>>('get_embedding').asFunction();
}
