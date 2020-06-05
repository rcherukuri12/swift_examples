import Foundation
import TensorFlow
import PythonKit
import TensorBoardS
import struct readers.csv_reader

let logdir = URL(fileURLWithPath: "/home/raja/swift-jupyter/swift_examples/fourth/logs")
try? FileManager.default.removeItem(at: logdir)

// MARK: - Create SummaryWriter
let writer = try SummaryWriter(logdir: logdir,flushInterval: 5)

//input data loader
func load_processed_data(path:String,label:String)-> (Tensor<Float>,Tensor<Int32>) {
    var input = csv_reader(filePath:path)
    let Z = input.load_data(label:label)
    // we have to take the label indicies, convert it from Int->Int32 and shape it.
    let y = Tensor(ShapedArray(shape:[Z.1.count],scalars: Z.1.map{Int32($0)}))
    // let us flatten the X -array before re-shaping as we want.
    let flattened = Z.0.flatMap { $0 }
    let x = Tensor(ShapedArray(shape:[Z.0.count,4],scalars:flattened))
    return (x,y)
}

// defining a simple model
let hiddenSize: Int = 10
extension Dense{
   var bad_weights_count : Int {return self.weight.flattened().shape[0] - self.weight.nonZeroIndices().shape[0] }
}
struct Model: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer4 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer5 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer6 = Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: identity)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3, layer4, layer5, layer6)
    }
}



// intializing an optimizer
var classifier = Model()

let optimizer = SGD(for: classifier, learningRate: 0.02)
Context.local.learningPhase = .training

let train = load_processed_data(path:"/home/raja/data/iris/iris_training.csv",label:"species")
let test  = load_processed_data(path:"/home/raja/data/iris/iris_test.csv",label:"species")

func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}
// var writer = TensorFlowModels.CheckpointWriter()

// 1. Make use of methods on Differentiable or Layer that produce a backpropagation function.
// 2. Compose your derivative computation.


for i in 0..<10000 {
    let (ŷ, backprop) = classifier.appliedForBackpropagation(to: train.0)
    let (loss, 𝛁ŷ) = valueWithGradient(at: ŷ) { ŷ in softmaxCrossEntropy(logits: ŷ, labels: train.1) }
    
    let grad = 𝛁ŷ
    let g0 = Float(grad[0][0])!
    let g1 = Float(grad[0][1])!
    let g2 = Float(grad[0][2])!
    //print(grad[0])
    //print(grad[0][0])
    //print(grad[0][1])
    //print(grad[0][2])
    // 𝛁ŷ
    guard let floss = loss.scalar else{ break }
    let acc = accuracy(predictions: ŷ.argmax(squeezingAxis: 1), truths: train.1)
    writer.addScalar(tag: "training loss", scalar: floss, step: i)
    writer.addScalar(tag: "training accuracy", scalar: acc, step: i)
    if i % 50 == 0 {
    writer.addScalar(tag: "gradients", scalar:g0, step: i)
    writer.addScalar(tag: "gradients", scalar:g1, step: i)
    writer.addScalar(tag: "gradients", scalar:g2, step: i)
    }
    //writer.addScalar(tag: "grad_2", scalar:g2, step: i)
    //print("Method 2 : Model output: \(ŷ), Loss: \(loss)")
    //print("prediction: \(ŷ)")
    print("\(i) : \(loss)")
    let (𝛁model, _) = backprop(𝛁ŷ)
    optimizer.update(&classifier, along: 𝛁model)
}
//print(loss_array)
writer.flush()
writer.close()

let logits = classifier(test.0)
let predictions = logits.argmax(squeezingAxis: 1)
print("Test  accuracy: \(accuracy(predictions: predictions, truths: test.1))")
print(predictions)
print(test.1)
/* print(classifier.layer1.weight)
print(classifier.layer2.weight)
print(classifier.layer3.weight)
print(classifier.layer4.weight)
print(classifier.layer5.weight)
print(classifier.layer6.weight) */
print(classifier.layer1.bad_weights_count)
print(classifier.layer2.bad_weights_count)
print(classifier.layer3.bad_weights_count)
print(classifier.layer4.bad_weights_count)
print(classifier.layer5.bad_weights_count)
print(classifier.layer6.bad_weights_count)

