import TensorFlow

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

var input = csv_reader(filePath:"/home/raja/data/iris/iris.csv")
let Z = input.load_data(label:"species")
// we have to take the label indicies, convert it from Int->Int32 and shape it.
let y = Tensor(ShapedArray(shape:[Z.1.count],scalars: Z.1.map{Int32($0)}))
// let us flatten the X -array before re-shaping as we want.
let flattened = Z.0.flatMap { $0 }
let x = Tensor(ShapedArray(shape:[Z.0.count,4],scalars:flattened))

// 1. Make use of methods on Differentiable or Layer that produce a backpropagation function.
// 2. Compose your derivative computation.
var loss_array:Array<Float> = []
for _ in 0..<1000 {
    let (ŷ, backprop) = classifier.appliedForBackpropagation(to: x)
    let (loss, 𝛁ŷ) = valueWithGradient(at: ŷ) { ŷ in softmaxCrossEntropy(logits: ŷ, labels: y) }
    
    guard let floss = loss.scalar else{ break }
    loss_array.append(Float(floss))
    //print("Method 2 : Model output: \(ŷ), Loss: \(loss)")
    //print("prediction: \(ŷ)")
    let (𝛁model, _) = backprop(𝛁ŷ)
    optimizer.update(&classifier, along: 𝛁model)
}
print(loss_array)
