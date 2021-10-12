/**
 *
 * Sample code for SNU Radiology Lecture
 *
 * Created by JoonNyung Heo (12 Oct 2021)
 * https://jnheo.com
 * jnheo@jnheo.com
 *
 *
 * Code used here is mostly referenced from example code from  tflite_flutter_helper plugin
 * https://github.com/am15h/tflite_flutter_helper
 *
 * Copyright 2021, JoonNyung Heo, All rights reserved
 * Licensed under the Apache License, Version 2.0.
 *
 */
import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:collection/collection.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:file_picker/file_picker.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  double prediction = 0.0;
  File? _image;

  late Interpreter interpreter;
  late List<int> _inputShape;
  late List<int> _outputShape;

  late TensorImage _inputImage;
  late TensorBuffer _outputBuffer;

  late TfLiteType _inputType;
  late TfLiteType _outputType;

  @override
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(127.5, 127.5);

  @override
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  loadModel() async {
    interpreter = await Interpreter.fromAsset('model.tflite');
    _inputShape = interpreter.getInputTensor(0).shape;
    _outputShape = interpreter.getOutputTensor(0).shape;
    _inputType = interpreter.getInputTensor(0).type;
    _outputType = interpreter.getOutputTensor(0).type;

    _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);
    print("Model loaded");
  }

  pickImage() async {
    /* we could use image picker instead */
    FilePickerResult? result = await FilePicker.platform.pickFiles();

    if (result != null) {
      setState(() {
        // save selected file path to image variable
        // for display in widget
        _image = File(result.files.single.path!);
      });

      classifyImage(_image!);
    } else {
      // User canceled the picker
    }
  }

  TensorImage _preProcess() {
    int cropSize = min(_inputImage.height, _inputImage.width);
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
            _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        // .add(preProcessNormalizeOp) ## normalization needed if done during training
        .build()
        .process(_inputImage);
  }

  classifyImage(File image) async {
    // preprocess image and get prediction results

    _inputImage = TensorImage(_inputType);
    img.Image imageInput = img.decodeImage(image.readAsBytesSync())!;
    _inputImage.loadImage(imageInput);
    _inputImage = _preProcess();
    interpreter.run(_inputImage.buffer, _outputBuffer.getBuffer());

    double result = _outputBuffer.buffer.asFloat32List().first;

    setState(() {
      prediction = result;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            _image != null
                ? Image.file(File(_image!.path))
                : const Icon(Icons.image),
            Text("Prediction result : $prediction"),
            TextButton(
              onPressed: () {
                pickImage();
              },
              child: const Text("Choose image"),
            ),
          ],
        ),
      ),
    );
  }
}
