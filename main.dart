// flutter run -d chrome --no-sound-null-safety
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  TextEditingController controller_tamil = TextEditingController();
  String eng_sen = '';
  String tam_sen = '';

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.greenAccent,
          centerTitle: true,
          title: const Text('Language Translation'),
          titleTextStyle: TextStyle(color: Colors.deepPurple, fontSize: 30),
        ),
        body: Center(
          child: Container(
            // decoration: BoxDecoration(
            //     gradient: LinearGradient(
            //         begin: Alignment.topRight,
            //         end: Alignment.bottomLeft,
            //         colors: [Colors.blue, Colors.lightBlueAccent])),
            height: 500,
            width: 500,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(height: 100),
                const Text(
                  'Enter Tamil Sentence:',
                  style: TextStyle(fontSize: 25,color: Colors.black),
                ),
                Container(height: 20),
                TextField(
                    decoration: const InputDecoration(
                      hintText: 'Enter Tamil Sentence',
                      // labelText: 'Tamil Sentence',
                      // labelStyle: TextStyle(
                      //   fontSize: 20.0,
                      // ),
                    ),
                    controller: controller_tamil),
                Container(height: 40),
                const Text(
                  'Translated English Sentence:',
                  style: TextStyle(fontSize: 25),
                ),
                Container(height: 10),
                Container(
                    height: 30,
                    width: 500,
                    child: Text(
                      eng_sen,
                      style: TextStyle(fontSize: 25),
                    )),
                Container(height: 40),
                Center(
                    child: ElevatedButton(
                        onPressed: () async {
                          // final response = await http.get(Uri.parse('http://127.0.0.1:5000/user'));
                          // print(response.body);
                          final send = await http.post(
                              Uri.parse('http://127.0.0.1:5000'),
                              body:
                                  json.encode({'sen': controller_tamil.text}));
                          final response = await http
                              .get(Uri.parse('http://127.0.0.1:5000'));
                          final decoded = json.decode(response.body)
                              as Map<String, dynamic>;

                          setState(() {
                            eng_sen = decoded['greetings'];
                          });
                        },
                        child: const Text('Translate'))),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
