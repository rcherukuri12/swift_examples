import TensorFlow
import PythonKit
import struct readers.csv_reader


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

struct Model: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: identity)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
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

// 1. Make use of methods on Differentiable or Layer that produce a backpropagation function.
// 2. Compose your derivative computation.
//var checkpoint = TensorFlowCheckpointReader(checkpointPath: "ckp")
var loss_array: [Float] = []
var trainAccuracy: [Float] = []
for _ in 0..<20000 {
    let (ŷ, backprop) = classifier.appliedForBackpropagation(to: train.0)
    let (loss, 𝛁ŷ) = valueWithGradient(at: ŷ) { ŷ in softmaxCrossEntropy(logits: ŷ, labels: train.1) }
    
    guard let floss = loss.scalar else{ break }
    loss_array.append(Float(floss))
    let acc = accuracy(predictions: ŷ.argmax(squeezingAxis: 1), truths: train.1)
    trainAccuracy.append(acc)
    //print("Method 2 : Model output: \(ŷ), Loss: \(loss)")
    //print("prediction: \(ŷ)")
    print("\(loss)")
    let (𝛁model, _) = backprop(𝛁ŷ)
    optimizer.update(&classifier, along: 𝛁model)
}
//print(loss_array)

let plt = Python.import("matplotlib.pyplot")
plt.plot(loss_array)
plt.plot(trainAccuracy)
plt.title("train loss and accuracy")
plt.savefig("train_accuracy.png")

let logits = classifier(test.0)
let predictions = logits.argmax(squeezingAxis: 1)
print("Test  accuracy: \(accuracy(predictions: predictions, truths: test.1))")
print(predictions)
print(test.1)

/* func writeCheckpoint(to location: URL, name: String) throws {
    var tensors = [String: Tensor<Float>]()
    recursivelyObtainTensors(classifier, scope: "model", tensors: &tensors, separator: "/")
    let writer = CheckpointWriter(tensors: tensors)
    try writer.write(to: location, name: name)
}
 */

