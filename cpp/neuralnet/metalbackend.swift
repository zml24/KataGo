import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// A class that handles output to standard error.
class StandardError: TextOutputStream {
    /// Outputs the specified string to the standard error stream.
    func write(_ string: String) {
        /// Tries to write the UTF-8 encoded contents of the string to the standard error file handle.
        try? FileHandle.standardError.write(contentsOf: Data(string.utf8))
    }
}

/// A function to print error messages
func printError(_ item: Any) {
    // Create an instance of StandardError to direct output to the standard error stream
    var instance = StandardError()
    // Output the provided item to the standard error using the created instance
    print(item, to: &instance)
}

/// An extension to the Data struct for handling float data with optional FP16 conversion.
extension Data {
    /// Initializes a new Data instance using an UnsafeMutablePointer<Float32>, with optional conversion to FP16 format.
    /// - Parameters:
    ///   - floatsNoCopy: An UnsafeMutablePointer<Float32> containing the float data.
    ///   - shape: An array of NSNumber objects representing the shape of the data.
    init(
        floatsNoCopy: UnsafeMutablePointer<Float32>,
        shape: [NSNumber]
    ) {
        self.init(
            bytesNoCopy: floatsNoCopy,
            count: shape.countBytesOfFloat32(),
            deallocator: .none)
    }
}

/// Extension to MPSNDArray to convert from MPSGraphTensor, and to read/write bytes from/to UnsafeMutableRawPointer
extension MPSNDArray {
    /// Read bytes from the buffer
    /// - Parameter buffer: The buffer to read
    func readBytes(_ buffer: UnsafeMutableRawPointer) {
        self.readBytes(buffer, strideBytes: nil)
    }

    /// Write bytes to the buffer
    /// - Parameter buffer: The buffer to write
    func writeBytes(_ buffer: UnsafeMutableRawPointer) {
        self.writeBytes(buffer, strideBytes: nil)
    }
}

private func castTensorIfNeeded(
    graph: MPSGraph,
    sourceTensor: MPSGraphTensor,
    to dataType: MPSDataType
) -> MPSGraphTensor {
    if sourceTensor.dataType == dataType {
        return sourceTensor
    }
    return graph.cast(sourceTensor, to: dataType, name: nil)
}

/// Extension to Array to count number of elements and bytes
extension Array where Element == NSNumber {
    /// Count number of elements
    /// - Returns: Number of elements
    func countElements() -> Int {
        return reduce(1, { $0 * $1.intValue })
    }

    /// Count number of bytes
    /// - Parameter dataType: The data type
    /// - Returns: Number of bytes
    func countBytesOfFloat32() -> Int {
        return countElements() * MemoryLayout<Float32>.size
    }
}

/// Extension to MPSGraph to the mish activation function
extension MPSGraph {
    /// This function applies the Mish activation function on the input tensor `x`. The Mish function is defined as
    /// x * tanh(Softplus(x)), where Softplus(x) is defined as log(1 + exp(min(x, 10.39))) if x < 10.39 and x otherwise.
    /// When FP16 is later used, the threshold of softplus will need to be modified to 10.39, which is different from
    /// the original 20. This is because exp(10.39) = 32532.666936 < 32767.0 < 65504.0, so the result of exp(10.39) can
    /// be represented by float16. If the threshold of softplus is 20, the result of exp(20) is 485165195.40979004,
    /// which is out of range of float16.
    /// - Parameter tensor: The input tensor of mish activation function
    /// - Returns: The output tensor of mish activation function
    func mish(tensor: MPSGraphTensor) -> MPSGraphTensor {
        assert(tensor.dataType == .float32)

        let one = 1.0
        let threshold = 20.0
        let thresholdTensor = constant(threshold, dataType: tensor.dataType)
        let minimumTensor = minimum(tensor, thresholdTensor, name: nil)
        let expTensor = exponent(with: minimumTensor, name: nil)
        let oneTensor = constant(one, dataType: tensor.dataType)
        let addTensor = addition(expTensor, oneTensor, name: nil)
        let logTensor = logarithm(with: addTensor, name: nil)
        let lessTensor = lessThan(tensor, thresholdTensor, name: nil)
        let selectTensor = select(
            predicate: lessTensor, trueTensor: logTensor, falseTensor: tensor, name: nil)
        let tanhTensor = tanh(with: selectTensor, name: nil)
        let mulTensor = multiplication(tensor, tanhTensor, name: nil)

        return mulTensor
    }
}

/// A structure that represents the input shape
struct InputShape {
    /// Create a shape for the input tensor
    /// - Parameters:
    ///   - batchSize: Batch size
    ///   - numChannels: Number of channels
    ///   - nnYLen: Y length
    ///   - nnXLen: X length
    /// - Returns: The shape
    static func create(
        batchSize: NSNumber,
        numChannels: NSNumber,
        nnYLen: NSNumber,
        nnXLen: NSNumber
    ) -> [NSNumber] {
        let shape = [
            batchSize,
            numChannels,
            nnYLen,
            nnXLen,
        ]
        return shape
    }

    /// Get the channel axis
    /// - Returns: The channel axis
    static func getChannelAxis() -> Int {
        return 1
    }

    /// Get the HW axes
    /// - Returns: The HW axes
    static func getHWAxes() -> [NSNumber] {
        let hwAxes = [2, 3] as [NSNumber]
        return hwAxes
    }
}

/// A structure that represents the input layer
struct InputLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    /// Initialize a InputLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    ///   - numChannels: Number of channels
    ///   - dataType: Data type
    init(
        graph: MPSGraph,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        numChannels: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        shape = InputShape.create(
            batchSize: -1,
            numChannels: numChannels,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        self.tensor = graph.placeholder(
            shape: shape,
            dataType: dataType,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents an input global layer for a neural network model.
struct InputGlobalLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    /// Initializes an InputGlobalLayer object with a graph, batch size, number of global features, data type, and input shape.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - numGlobalFeatures: The number of global features.
    ///   - dataType: The data type.
    init(
        graph: MPSGraph,
        numGlobalFeatures: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        shape = InputShape.create(
            batchSize: -1,
            numChannels: numGlobalFeatures,
            nnYLen: 1,
            nnXLen: 1)

        self.tensor = graph.placeholder(
            shape: shape,
            dataType: dataType,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure representing the input meta layer for a neural network graph.
struct InputMetaLayer {
    /// A `MPSGraphTensor` representing the placeholder tensor in the graph.
    let tensor: MPSGraphTensor
    /// An array of `NSNumber` representing the shape of the tensor placeholder.
    let shape: [NSNumber]

    /// Initializes a new `InputMetaLayer` instance with the given graph and number of meta features.
    ///
    /// - Parameters:
    ///   - graph: The `MPSGraph` instance where the placeholder tensor will be created.
    ///   - numMetaFeatures: The number of meta features (channels) for the input tensor.
    ///   - dataType: The data type
    ///
    /// This initializer sets the shape of the input tensor using a helper function `InputShape.create` with
    /// a dynamic batch size (-1), the specified number of channels, and a spatial size of 1x1 (nnYLen and nnXLen).
    /// It also creates a placeholder tensor in the MPS graph with the specified shape and data type `float32`.
    init(
        graph: MPSGraph,
        numMetaFeatures: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        // Define the shape of the input tensor with dynamic batch size, specified number of channels, and spatial dimensions 1x1.
        shape = InputShape.create(
            batchSize: -1,
            numChannels: numMetaFeatures,
            nnYLen: 1,
            nnXLen: 1)

        // Create a placeholder tensor in the graph with the above-defined shape and data type float32.
        self.tensor = graph.placeholder(
            shape: shape,
            dataType: dataType,
            name: nil)
    }
}

/// A structure that represents a mask layer for a neural network model.
struct MaskLayer {
    let tensor: MPSGraphTensor
    let shape: [NSNumber]

    /// Initializes a MaskLayer object with a graph, batch size, x and y lengths, data type, and input shape.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - nnXLen: The length of the x-axis.
    ///   - nnYLen: The length of the y-axis.
    ///   - dataType: The data type.
    init(
        graph: MPSGraph,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        dataType: MPSDataType = .float32
    ) {
        shape = InputShape.create(
            batchSize: -1,
            numChannels: 1,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        self.tensor = graph.placeholder(
            shape: shape,
            dataType: dataType,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents a layer which performs the summation operation on a mask layer.
struct MaskSumLayer {
    let tensor: MPSGraphTensor

    /// Initializes a MaskSumLayer object with a given tensor.
    /// - Parameter tensor: The tensor to use for the layer.
    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    /// Initializes a MaskSumLayer object with a graph, a mask layer, and a boolean flag indicating whether to use NHWC or NCHW format.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - maskTensor: The mask tensor.
    init(
        graph: MPSGraph,
        maskTensor: MPSGraphTensor
    ) {
        let hwAxes = InputShape.getHWAxes()

        self.tensor = graph.reductionSum(
            with: maskTensor,
            axes: hwAxes,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents a layer which performs square root, subtraction, and multiplication operations on a MaskSumLayer object.
struct MaskSumSqrtS14M01Layer {
    let tensor: MPSGraphTensor

    /// Initializes a MaskSumSqrtS14M01Layer object with a given tensor.
    /// - Parameter tensor: The tensor to use for the layer.
    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    /// Initializes a MaskSumSqrtS14M01Layer object with a graph, a MaskSumLayer object, and a boolean flag indicating whether to use 16-bit floating-point data type.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - maskSum: The MaskSumLayer object.
    init(
        graph: MPSGraph,
        maskSum: MaskSumLayer
    ) {
        let sqrtMaskSum = graph.squareRoot(with: maskSum.tensor, name: nil)

        let fourTeen = graph.constant(
            14.0,
            shape: [1],
            dataType: maskSum.tensor.dataType)

        let subtracted = graph.subtraction(sqrtMaskSum, fourTeen, name: nil)

        let zeroPointone = graph.constant(
            0.1,
            shape: [1],
            dataType: maskSum.tensor.dataType)

        self.tensor = graph.multiplication(
            subtracted,
            zeroPointone,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A structure that represents a layer which performs squaring and subtraction operations on a MaskSumSqrtS14M01Layer object.
struct MaskSumSqrtS14M01SquareS01Layer {
    let tensor: MPSGraphTensor

    /// Initializes a MaskSumSqrtS14M01SquareS01Layer object with a given tensor.
    /// - Parameter tensor: The tensor to use for the layer.
    init(tensor: MPSGraphTensor) {
        self.tensor = tensor
        assert(self.tensor.shape?.count == 4)
    }

    /// Initializes a MaskSumSqrtS14M01SquareS01Layer object with a graph, a MaskSumSqrtS14M01Layer object, and a boolean flag indicating whether to use 16-bit floating-point data type.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - maskSumSqrtS14M01: The MaskSumSqrtS14M01Layer object.
    init(
        graph: MPSGraph,
        maskSumSqrtS14M01: MaskSumSqrtS14M01Layer
    ) {
        let squared = graph.square(with: maskSumSqrtS14M01.tensor, name: nil)

        let zeroPointone = graph.constant(
            0.1,
            shape: [1],
            dataType: maskSumSqrtS14M01.tensor.dataType)

        self.tensor = graph.subtraction(
            squared,
            zeroPointone,
            name: nil)

        assert(self.tensor.shape?.count == 4)
    }
}

/// A Swift structure that represents a network tester, which tests various neural network configurations.
struct NetworkTester {

    /// A static function that tests a custom neural network configuration with the given parameters.
    /// - Parameters:
    ///   - batchSize: The number of input batches.
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    ///   - numChannels: The number of channels in the input tensor.
    ///   - input: A pointer to the input data.
    ///   - mask: A pointer to the mask data.
    ///   - output: A pointer to the output data.
    ///   - networkBuilder: A closure that takes an MPSGraph, InputLayer, and MaskLayer, and returns an MPSGraphTensor representing the custom network configuration.
    static func test(
        batchSize: NSNumber,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        numChannels: NSNumber,
        input: UnsafeMutablePointer<Float32>,
        mask: UnsafeMutablePointer<Float32>,
        output: UnsafeMutablePointer<Float32>,
        networkBuilder: (MPSGraph, InputLayer, MaskLayer) -> MPSGraphTensor
    ) {

        // Create a Metal device.
        let device = MTLCreateSystemDefaultDevice()!

        // Create a MPSGraph.
        let graph = MPSGraph()

        // Create the input and mask layers.
        let inputLayer = InputLayer(
            graph: graph,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: numChannels)

        let maskLayer = MaskLayer(
            graph: graph,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        // Build the custom network configuration using the provided networkBuilder closure.
        let resultTensor = networkBuilder(graph, inputLayer, maskLayer)

        // Create input shape
        let inputShape = InputShape.create(
            batchSize: batchSize,
            numChannels: numChannels,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        // Create MPSNDArrayDescriptors from the input shape.
        let sourceDescriptor = MPSNDArrayDescriptor(
            dataType: inputLayer.tensor.dataType,
            shape: inputShape)

        // Create MPSNDArray from the source descriptor.
        let sourceArray = MPSNDArray(
            device: device,
            descriptor: sourceDescriptor)

        // Create a mask shape
        let maskShape = InputShape.create(
            batchSize: batchSize,
            numChannels: 1,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        // Create MPSNDArrayDescriptors from the mask shape.
        let maskDescriptor = MPSNDArrayDescriptor(
            dataType: maskLayer.tensor.dataType,
            shape: maskShape)

        // Create MPSNDArray from the mask descriptor.
        let maskArray = MPSNDArray(
            device: device,
            descriptor: maskDescriptor)

        // Write input and mask data to their respective MPSNDArrays, converting to FP16 if necessary.
        sourceArray.writeBytes(input)
        maskArray.writeBytes(mask)

        // Create MPSGraphTensorData objects from the source and mask arrays.
        let sourceTensorData = MPSGraphTensorData(sourceArray)
        let maskTensorData = MPSGraphTensorData(maskArray)

        // Execute the graph and fetch the result.
        let fetch = graph.run(
            feeds: [
                inputLayer.tensor: sourceTensorData,
                maskLayer.tensor: maskTensorData,
            ],
            targetTensors: [resultTensor],
            targetOperations: nil)

        // Read the output data from the result tensor, converting from FP16 to FP32 if necessary.
        fetch[resultTensor]?.mpsndarray().readBytes(output)
    }
}

/// A struct that represents a description of convolutional layer.
public struct SWConvLayerDesc {
    let convYSize: NSNumber
    let convXSize: NSNumber
    let inChannels: NSNumber
    let outChannels: NSNumber
    let dilationY: Int
    let dilationX: Int
    let weights: UnsafeMutablePointer<Float32>

    /// Initializes a SWConvLayerDesc object.
    /// - Parameters:
    ///   - convYSize: The Y size of the convolution.
    ///   - convXSize: The X size of the convolution.
    ///   - inChannels: The number of input channels.
    ///   - outChannels: The number of output channels.
    ///   - dilationY: The dilation in the Y direction.
    ///   - dilationX: The dilation in the X direction.
    ///   - weights: A pointer to the weights.
    init(
        convYSize: NSNumber,
        convXSize: NSNumber,
        inChannels: NSNumber,
        outChannels: NSNumber,
        dilationY: Int,
        dilationX: Int,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.convYSize = convYSize
        self.convXSize = convXSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dilationY = dilationY
        self.dilationX = dilationX
        self.weights = weights
    }
}

public func createSWConvLayerDesc(
    convYSize: Int32,
    convXSize: Int32,
    inChannels: Int32,
    outChannels: Int32,
    dilationY: Int32,
    dilationX: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWConvLayerDesc {
    return SWConvLayerDesc(
        convYSize: convYSize as NSNumber,
        convXSize: convXSize as NSNumber,
        inChannels: inChannels as NSNumber,
        outChannels: outChannels as NSNumber,
        dilationY: Int(dilationY),
        dilationX: Int(dilationX),
        weights: weights)
}

/// A class that represents a convolutional layer using MPSGraph
class ConvLayer {
    /// The result tensor of the convolutional operation
    let resultTensor: MPSGraphTensor
    /// The convolution 2D operation descriptor
    let convDescriptor = MPSGraphConvolution2DOpDescriptor(
        strideInX: 1,
        strideInY: 1,
        dilationRateInX: 1,
        dilationRateInY: 1,
        groups: 1,
        paddingStyle: .TF_SAME,
        dataLayout: .NCHW,
        weightsLayout: .OIHW)!

    /// Class method that tests the convolutional layer by running a forward pass
    /// - Parameters:
    ///   - descriptor: A descriptor for the convolutional layer
    ///   - nnXLen: The width of the input tensor
    ///   - nnYLen: The height of the input tensor
    ///   - batchSize: The batch size of the input tensor
    ///   - input: A pointer to the input tensor data
    ///   - output: A pointer to the output tensor data
    class func test(
        descriptor: SWConvLayerDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        batchSize: NSNumber,
        input: UnsafeMutablePointer<Float32>,
        output: UnsafeMutablePointer<Float32>
    ) {
        let device = MTLCreateSystemDefaultDevice()!
        let graph = MPSGraph()

        let source = InputLayer(
            graph: graph,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.inChannels)

        let conv = ConvLayer(
            graph: graph,
            sourceTensor: source.tensor,
            descriptor: descriptor,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let inputShape = InputShape.create(
            batchSize: batchSize,
            numChannels: descriptor.inChannels,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        let sourceDescriptor = MPSNDArrayDescriptor(
            dataType: source.tensor.dataType,
            shape: inputShape)

        let sourceArray = MPSNDArray(
            device: device,
            descriptor: sourceDescriptor)

        sourceArray.writeBytes(input)
        let sourceTensorData = MPSGraphTensorData(sourceArray)

        let fetch = graph.run(
            feeds: [source.tensor: sourceTensorData],
            targetTensors: [conv.resultTensor],
            targetOperations: nil)

        fetch[conv.resultTensor]?.mpsndarray().readBytes(output)
    }

    /// Initializes a ConvLayer object
    /// - Parameters:
    ///   - graph: An MPSGraph object
    ///   - sourceTensor: The input tensor for the convolutional layer
    ///   - descriptor: A descriptor for the convolutional layer
    ///   - nnXLen: The width of the input tensor
    ///   - nnYLen: The height of the input tensor
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        descriptor: SWConvLayerDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        computationDataType: MPSDataType? = nil,
        outputDataType: MPSDataType? = nil
    ) {
        let opDataType = computationDataType ?? sourceTensor.dataType
        let weightsShape = [
            descriptor.outChannels,
            descriptor.inChannels,
            descriptor.convYSize,
            descriptor.convXSize,
        ]

        let weightsTensor = makeConstantTensor(
            graph: graph,
            pointer: descriptor.weights,
            shape: weightsShape,
            dataType: opDataType)

        let conv = graph.convolution2D(
            castTensorIfNeeded(graph: graph, sourceTensor: sourceTensor, to: opDataType),
            weights: weightsTensor,
            descriptor: convDescriptor,
            name: nil)
        resultTensor = castTensorIfNeeded(
            graph: graph,
            sourceTensor: conv,
            to: outputDataType ?? sourceTensor.dataType)

        assert(resultTensor.shape?.count == 4)
    }
}

public func testConvLayer(
    descriptor: SWConvLayerDesc,
    nnXLen: Int32,
    nnYLen: Int32,
    batchSize: Int32,
    input: UnsafeMutablePointer<Float32>,
    output: UnsafeMutablePointer<Float32>
) {
    ConvLayer.test(
        descriptor: descriptor,
        nnXLen: nnXLen as NSNumber,
        nnYLen: nnYLen as NSNumber,
        batchSize: batchSize as NSNumber,
        input: input,
        output: output)
}

/// A struct that represents a description of a batch normalization layer.
public struct SWBatchNormLayerDesc {
    let numChannels: NSNumber
    let mergedScale: UnsafeMutablePointer<Float32>
    let mergedBias: UnsafeMutablePointer<Float32>

    /// Initializes a SWBatchNormLayerDesc object.
    /// - Parameters:
    ///   - numChannels: The number of channels in the input tensor.
    ///   - mergedScale: A pointer to the merged scale.
    ///   - mergedBias: A pointer to the merged bias.
    init(
        numChannels: NSNumber,
        mergedScale: UnsafeMutablePointer<Float32>,
        mergedBias: UnsafeMutablePointer<Float32>
    ) {
        self.numChannels = numChannels
        self.mergedScale = mergedScale
        self.mergedBias = mergedBias
    }
}

public func createSWBatchNormLayerDesc(
    numChannels: Int32,
    mergedScale: UnsafeMutablePointer<Float32>,
    mergedBias: UnsafeMutablePointer<Float32>
) -> SWBatchNormLayerDesc {
    return SWBatchNormLayerDesc(
        numChannels: numChannels as NSNumber,
        mergedScale: mergedScale,
        mergedBias: mergedBias)
}

/// A class that represents a batch normalization layer.
class BatchNormLayer {
    let resultTensor: MPSGraphTensor

    /// Executes a test for the batch normalization layer.
    /// - Parameters:
    ///   - descriptor: The description of the batch normalization layer.
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    ///   - batchSize: The number of input batches.
    ///   - input: A pointer to the input data.
    ///   - mask: A pointer to the mask data.
    ///   - output: A pointer to the output data.
    class func test(
        descriptor: SWBatchNormLayerDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        batchSize: NSNumber,
        input: UnsafeMutablePointer<Float32>,
        mask: UnsafeMutablePointer<Float32>,
        output: UnsafeMutablePointer<Float32>
    ) {

        NetworkTester.test(
            batchSize: batchSize,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.numChannels,
            input: input,
            mask: mask,
            output: output
        ) { graph, inputLayer, maskLayer in

            let batchNorm = BatchNormLayer(
                graph: graph,
                sourceTensor: inputLayer.tensor,
                maskTensor: maskLayer.tensor,
                descriptor: descriptor,
                nnXLen: nnXLen,
                nnYLen: nnYLen)

            return batchNorm.resultTensor
        }
    }

    /// Initializes a BatchNormLayer object with the specified parameters, and computes the normalized and masked result tensor.
    /// - Parameters:
    ///   - graph: The MPSGraph object used to build the BatchNormLayer.
    ///   - sourceTensor: The input tensor to the BatchNormLayer.
    ///   - maskTensor: The mask tensor to apply to the normalized tensor.
    ///   - descriptor: The BatchNormLayer descriptor containing parameters such as the number of channels, mean, variance, scale, and bias.
    ///   - nnXLen: The length of the input tensor in the X direction.
    ///   - nnYLen: The length of the input tensor in the Y direction.
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        descriptor: SWBatchNormLayerDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {
        let scaleBiasShape = InputShape.create(
            batchSize: 1,
            numChannels: descriptor.numChannels,
            nnYLen: 1,
            nnXLen: 1)

        let scaleTensor = makeConstantTensor(
            graph: graph,
            pointer: descriptor.mergedScale,
            shape: scaleBiasShape,
            dataType: sourceTensor.dataType)

        let biasTensor = makeConstantTensor(
            graph: graph,
            pointer: descriptor.mergedBias,
            shape: scaleBiasShape,
            dataType: sourceTensor.dataType)

        let scaled = graph.multiplication(
            sourceTensor,
            scaleTensor,
            name: nil)

        let normalized = graph.addition(
            scaled,
            biasTensor,
            name: nil)

        resultTensor = graph.multiplication(
            normalized,
            maskTensor,
            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

public func testBatchNormLayer(
    descriptor: SWBatchNormLayerDesc,
    nnXLen: Int32,
    nnYLen: Int32,
    batchSize: Int32,
    input: UnsafeMutablePointer<Float32>,
    mask: UnsafeMutablePointer<Float32>,
    output: UnsafeMutablePointer<Float32>
) {
    BatchNormLayer.test(
        descriptor: descriptor,
        nnXLen: nnXLen as NSNumber,
        nnYLen: nnYLen as NSNumber,
        batchSize: batchSize as NSNumber,
        input: input,
        mask: mask,
        output: output)
}

/// An enumeration of the different kinds of activation function.
public enum ActivationKind {
    case identity
    case relu
    case mish
}

/// A structure that represents an activation layer
struct ActivationLayer {
    let resultTensor: MPSGraphTensor

    /// Initialize an ActivationLayer object
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - activationKind: The activation kind
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        activationKind: ActivationKind
    ) {

        switch activationKind {
        case .relu:
            resultTensor = graph.reLU(with: sourceTensor, name: nil)
        case .mish:
            resultTensor = graph.mish(tensor: sourceTensor)
        default:
            resultTensor = sourceTensor
        }

        assert(resultTensor.shape == sourceTensor.shape)
    }
}

/// A class that represents a residual block in a convolutional neural network.
public class SWResidualBlockDesc: BlockDescriptor {
    /// A description of the batch normalization layer that is applied before the first convolutional layer.
    let preBN: SWBatchNormLayerDesc

    /// The type of activation function that is applied before the first convolutional layer.
    let preActivation: ActivationKind

    /// A description of the convolutional layer that is applied in the middle of the residual block.
    let regularConv: SWConvLayerDesc

    /// A description of the batch normalization layer that is applied after the middle convolutional layer.
    let midBN: SWBatchNormLayerDesc

    /// The type of activation function that is applied after the middle convolutional layer.
    let midActivation: ActivationKind

    /// A description of the convolutional layer that is applied at the end of the residual block.
    let finalConv: SWConvLayerDesc

    /// Initializes a `SWResidualBlockDesc` object.
    /// - Parameters:
    ///   - preBN: A description of the batch normalization layer that is applied before the first convolutional layer.
    ///   - preActivation: The type of activation function that is applied before the first convolutional layer.
    ///   - regularConv: A description of the convolutional layer that is applied in the middle of the residual block.
    ///   - midBN: A description of the batch normalization layer that is applied after the middle convolutional layer.
    ///   - midActivation: The type of activation function that is applied after the middle convolutional layer.
    ///   - finalConv: A description of the convolutional layer that is applied at the end of the residual block.
    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        regularConv: SWConvLayerDesc,
        midBN: SWBatchNormLayerDesc,
        midActivation: ActivationKind,
        finalConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }
}

public func createSWResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    regularConv: SWConvLayerDesc,
    midBN: SWBatchNormLayerDesc,
    midActivation: ActivationKind,
    finalConv: SWConvLayerDesc
) -> SWResidualBlockDesc {
    return SWResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        regularConv: regularConv,
        midBN: midBN,
        midActivation: midActivation,
        finalConv: finalConv)
}

/// A class that represents a Residual Block layer
class ResidualBlock {
    let resultTensor: MPSGraphTensor

    /// A function that runs tests on the Residual Block layer
    ///
    /// - Parameters:
    ///   - descriptor: The Residual Block descriptor
    ///   - batchSize: Batch size
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    ///   - input: The input float32 pointer
    ///   - mask: The mask float32 pointer
    ///   - output: The output float32 pointer
    class func test(
        descriptor: SWResidualBlockDesc,
        batchSize: NSNumber,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        input: UnsafeMutablePointer<Float32>,
        mask: UnsafeMutablePointer<Float32>,
        output: UnsafeMutablePointer<Float32>
    ) {

        NetworkTester.test(
            batchSize: batchSize,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.preBN.numChannels,
            input: input,
            mask: mask,
            output: output
        ) { graph, inputLayer, maskLayer in

            let block = ResidualBlock(
                graph: graph,
                sourceTensor: inputLayer.tensor,
                maskTensor: maskLayer.tensor,
                descriptor: descriptor,
                nnXLen: nnXLen,
                nnYLen: nnYLen)

            return block.resultTensor
        }
    }

    /// Initialize a ResidualBlock object
    ///
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - maskTensor: The mask tensor
    ///   - descriptor: The Residual Block descriptor
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        descriptor: SWResidualBlockDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {
        let preBN = BatchNormLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.preBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let preActivation = ActivationLayer(
            graph: graph,
            sourceTensor: preBN.resultTensor,
            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(
            graph: graph,
            sourceTensor: preActivation.resultTensor,
            descriptor: descriptor.regularConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let midBN = BatchNormLayer(
            graph: graph,
            sourceTensor: regularConv.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.midBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let midActivation = ActivationLayer(
            graph: graph,
            sourceTensor: midBN.resultTensor,
            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(
            graph: graph,
            sourceTensor: midActivation.resultTensor,
            descriptor: descriptor.finalConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        resultTensor = graph.addition(
            sourceTensor,
            finalConv.resultTensor,
            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

public func testResidualBlock(
    descriptor: SWResidualBlockDesc,
    batchSize: Int32,
    nnXLen: Int32,
    nnYLen: Int32,
    input: UnsafeMutablePointer<Float32>,
    mask: UnsafeMutablePointer<Float32>,
    output: UnsafeMutablePointer<Float32>
) {
    ResidualBlock.test(
        descriptor: descriptor,
        batchSize: batchSize as NSNumber,
        nnXLen: nnXLen as NSNumber,
        nnYLen: nnYLen as NSNumber,
        input: input,
        mask: mask,
        output: output)
}

/// A structure that represents a global pooling layer
struct GlobalPoolingLayer {
    /// The resulting tensor after applying the global pooling operation
    let resultTensor: MPSGraphTensor

    /// Initialize a GlobalPoolingLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - sourceTensor: The source tensor to be pooled
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The sum of the mask
    ///   - maskSumSqrtS14M01Tensor: The multiplication of subtraction of square root of the sum of the mask
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor
    ) {
        let hwAxes = InputShape.getHWAxes()
        let channelAxis = InputShape.getChannelAxis()

        let sumTensor = graph.reductionSum(
            with: sourceTensor,
            axes: hwAxes,
            name: nil)

        let meanTensor = graph.division(sumTensor, maskSumTensor, name: nil)

        let meanMaskTensor = graph.multiplication(
            meanTensor,
            maskSumSqrtS14M01Tensor,
            name: nil)

        let oneTensor = graph.constant(1.0, dataType: sourceTensor.dataType)
        let maskM1Tensor = graph.subtraction(maskTensor, oneTensor, name: nil)
        let addition = graph.addition(sourceTensor, maskM1Tensor, name: nil)

        let maxTensor = graph.reductionMaximum(
            with: addition,
            axes: hwAxes,
            name: nil)

        resultTensor = graph.concatTensors(
            [
                meanTensor,
                meanMaskTensor,
                maxTensor,
            ],
            dimension: channelAxis,
            name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(resultTensor.shape?[2] == 1)
        assert(resultTensor.shape?[3] == 1)
    }
}

/// A structure that represents a layer that performs global pooling on the input tensor
struct GlobalPoolingValueLayer {
    let resultTensor: MPSGraphTensor

    /// Initialize a GlobalPoolingValueLayer object
    /// - Parameters:
    ///   - graph: The graph
    ///   - sourceTensor: The input tensor
    ///   - maskSumTensor: The sum of the mask
    ///   - maskSumSqrtS14M01Tensor: The multiplication of subtraction of square root of the sum of the mask
    ///   - maskSumSqrtS14M01SquareS01Tensor: The subtraction of square of multiplication of subtraction of square root of the sum of the mask
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor
    ) {
        let hwAxes = InputShape.getHWAxes()
        let channelAxis = InputShape.getChannelAxis()

        let sumTensor = graph.reductionSum(
            with: sourceTensor,
            axes: hwAxes,
            name: nil)

        let meanTensor = graph.division(sumTensor, maskSumTensor, name: nil)

        let meanMaskTensor = graph.multiplication(
            meanTensor,
            maskSumSqrtS14M01Tensor,
            name: nil)

        let meanMaskSquareTensor = graph.multiplication(
            meanTensor,
            maskSumSqrtS14M01SquareS01Tensor,
            name: nil)

        resultTensor = graph.concatTensors(
            [
                meanTensor,
                meanMaskTensor,
                meanMaskSquareTensor,
            ],
            dimension: channelAxis,
            name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(resultTensor.shape?[2] == 1)
        assert(resultTensor.shape?[3] == 1)
    }
}

/// A struct that represents a matrix multiplication layer descriptor
public struct SWMatMulLayerDesc {
    /// The number of input channels
    let inChannels: NSNumber
    /// The number of output channels
    let outChannels: NSNumber
    /// The weights used for the matrix multiplication
    let weights: UnsafeMutablePointer<Float32>?

    /// Initialize a SWMatMulLayerDesc object
    /// - Parameters:
    ///   - inChannels: The number of input channels
    ///   - outChannels: The number of output channels
    ///   - weights: The weights used for the matrix multiplication
    init(
        inChannels: NSNumber,
        outChannels: NSNumber,
        weights: UnsafeMutablePointer<Float32>?
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }
}

public func createSWMatMulLayerDesc(
    inChannels: Int32,
    outChannels: Int32,
    weights: UnsafeMutablePointer<Float32>?
) -> SWMatMulLayerDesc {
    return SWMatMulLayerDesc(
        inChannels: inChannels as NSNumber,
        outChannels: outChannels as NSNumber,
        weights: weights)
}

/// A structure representing a matrix multiplication layer.
struct MatMulLayer {
    /// The resulting tensor from the layer.
    let resultTensor: MPSGraphTensor

    /// Initializes a MatMulLayer object.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - descriptor: The matrix multiplication layer descriptor.
    ///   - sourceTensor: The input tensor to the layer.
    init(
        graph: MPSGraph,
        descriptor: SWMatMulLayerDesc,
        sourceTensor: MPSGraphTensor
    ) {

        assert(
            (sourceTensor.shape?.count == 4) || (sourceTensor.shape?[1] == descriptor.inChannels))
        assert(
            (sourceTensor.shape?.count == 2) || (sourceTensor.shape?[1] == descriptor.inChannels))
        precondition(descriptor.weights != nil)

        let weightsShape = [
            descriptor.inChannels,
            descriptor.outChannels,
        ]

        let weightsData = Data(
            floatsNoCopy: descriptor.weights!,
            shape: weightsShape)

        let weightsTensor = graph.constant(
            weightsData,
            shape: weightsShape,
            dataType: sourceTensor.dataType)

        let shape = [-1, descriptor.inChannels]

        let reshapedSource = graph.reshape(
            sourceTensor,
            shape: shape,
            name: nil)

        resultTensor = graph.matrixMultiplication(
            primary: reshapedSource,
            secondary: weightsTensor,
            name: nil)

        assert(resultTensor.shape?.count == 2)
    }
}

/// An Objective-C class that represents the bias layer description used in Swift.
public struct SWMatBiasLayerDesc {
    /// The number of channels.
    let numChannels: NSNumber
    /// The pointer to the weights.
    let weights: UnsafeMutablePointer<Float32>

    /// Initialize an instance of SWMatBiasLayerDesc.
    /// - Parameters:
    ///   - numChannels: The number of channels.
    ///   - weights: The pointer to the weights.
    init(
        numChannels: NSNumber,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.numChannels = numChannels
        self.weights = weights
    }
}

public func createSWMatBiasLayerDesc(
    numChannels: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWMatBiasLayerDesc {
    return SWMatBiasLayerDesc(
        numChannels: numChannels as NSNumber,
        weights: weights)
}

/// A structure that performs matrix bias operations
struct MatBiasLayer {
    /// The resulting tensor from the layer.
    let resultTensor: MPSGraphTensor

    /// Initializes a MatBiasLayer object.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - descriptor: The descriptor that contains information about the layer
    ///   - sourceTensor: The input tensor to the layer.
    init(
        graph: MPSGraph,
        descriptor: SWMatBiasLayerDesc,
        sourceTensor: MPSGraphTensor
    ) {

        assert(
            (sourceTensor.shape?.count == 2) && (sourceTensor.shape?[1] == descriptor.numChannels))

        let weightsShape = [1, descriptor.numChannels]

        let weightsData = Data(
            floatsNoCopy: descriptor.weights,
            shape: weightsShape)

        let weightsTensor = graph.constant(
            weightsData,
            shape: weightsShape,
            dataType: sourceTensor.dataType)

        resultTensor = graph.addition(
            sourceTensor,
            weightsTensor,
            name: nil)
    }
}

/// A structure that performs bias operations in NC coordinates.
struct AddNCBiasLayer {
    /// The resulting tensor from the layer.
    let resultTensor: MPSGraphTensor

    /// Initializes an AddNCBiasLayer object.
    /// - Parameters:
    ///   - graph: The graph.
    ///   - sourceTensor: The input tensor to the layer.
    ///   - biasTensor: The bias tensor.
    ///   - nnXLen: The x length.
    ///   - nnYLen: The y length.
    ///   - numChannels: The number of channels.
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        biasTensor: MPSGraphTensor,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        numChannels: NSNumber
    ) {
        let shape = InputShape.create(
            batchSize: -1,
            numChannels: numChannels,
            nnYLen: 1,
            nnXLen: 1)

        assert(biasTensor.shape?[1] == shape[1])

        let reshaped = graph.reshape(biasTensor, shape: shape, name: nil)
        resultTensor = graph.addition(sourceTensor, reshaped, name: nil)

        assert(resultTensor.shape?.count == 4)
        assert(resultTensor.shape?[2] == nnYLen)
        assert(resultTensor.shape?[3] == nnXLen)
    }
}

/// A class that represents a residual block with global pooling.
public class SWGlobalPoolingResidualBlockDesc: BlockDescriptor {
    /// The batch normalization layer before the residual block.
    let preBN: SWBatchNormLayerDesc

    /// The pre-activation function of the residual block.
    let preActivation: ActivationKind

    /// The regular convolutional layer in the residual block.
    let regularConv: SWConvLayerDesc

    /// The convolutional layer for global pooling.
    let gpoolConv: SWConvLayerDesc

    /// The batch normalization layer after the global pooling convolutional layer.
    let gpoolBN: SWBatchNormLayerDesc

    /// The activation function after the global pooling batch normalization layer.
    let gpoolActivation: ActivationKind

    /// The matrix multiplication layer that multiplies the global pooled output with a bias.
    let gpoolToBiasMul: SWMatMulLayerDesc

    /// The batch normalization layer after the matrix multiplication layer.
    let midBN: SWBatchNormLayerDesc

    /// The activation function after the mid batch normalization layer.
    let midActivation: ActivationKind

    /// The final convolutional layer in the residual block.
    let finalConv: SWConvLayerDesc

    /// Initialize a SWGlobalPoolingResidualBlockDesc object.
    /// - Parameters:
    ///   - preBN: The batch normalization layer before the residual block.
    ///   - preActivation: The pre-activation function of the residual block.
    ///   - regularConv: The regular convolutional layer in the residual block.
    ///   - gpoolConv: The convolutional layer for global pooling.
    ///   - gpoolBN: The batch normalization layer after the global pooling convolutional layer.
    ///   - gpoolActivation: The activation function after the global pooling batch normalization layer.
    ///   - gpoolToBiasMul: The matrix multiplication layer that multiplies the global pooled output with a bias.
    ///   - midBN: The batch normalization layer after the matrix multiplication layer.
    ///   - midActivation: The activation function after the mid batch normalization layer.
    ///   - finalConv: The final convolutional layer in the residual block.
    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        regularConv: SWConvLayerDesc,
        gpoolConv: SWConvLayerDesc,
        gpoolBN: SWBatchNormLayerDesc,
        gpoolActivation: ActivationKind,
        gpoolToBiasMul: SWMatMulLayerDesc,
        midBN: SWBatchNormLayerDesc,
        midActivation: ActivationKind,
        finalConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.gpoolConv = gpoolConv
        self.gpoolBN = gpoolBN
        self.gpoolActivation = gpoolActivation
        self.gpoolToBiasMul = gpoolToBiasMul
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }
}

public func createSWGlobalPoolingResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    regularConv: SWConvLayerDesc,
    gpoolConv: SWConvLayerDesc,
    gpoolBN: SWBatchNormLayerDesc,
    gpoolActivation: ActivationKind,
    gpoolToBiasMul: SWMatMulLayerDesc,
    midBN: SWBatchNormLayerDesc,
    midActivation: ActivationKind,
    finalConv: SWConvLayerDesc
) -> SWGlobalPoolingResidualBlockDesc {

    return SWGlobalPoolingResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        regularConv: regularConv,
        gpoolConv: gpoolConv,
        gpoolBN: gpoolBN,
        gpoolActivation: gpoolActivation,
        gpoolToBiasMul: gpoolToBiasMul,
        midBN: midBN,
        midActivation: midActivation,
        finalConv: finalConv)
}

/// A class representing a residual block with global pooling
class GlobalPoolingResidualBlock {
    let resultTensor: MPSGraphTensor

    /// A method to test the global pooling residual block
    ///
    /// - Parameters:
    ///   - descriptor: The descriptor of the global pooling residual block
    ///   - batchSize: The batch size
    ///   - nnXLen: The X length
    ///   - nnYLen: The Y length
    ///   - input: The input pointer
    ///   - mask: The mask pointer
    ///   - output: The output pointer
    class func test(
        descriptor: SWGlobalPoolingResidualBlockDesc,
        batchSize: NSNumber,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        input: UnsafeMutablePointer<Float32>,
        mask: UnsafeMutablePointer<Float32>,
        output: UnsafeMutablePointer<Float32>
    ) {

        NetworkTester.test(
            batchSize: batchSize,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.preBN.numChannels,
            input: input,
            mask: mask,
            output: output
        ) { graph, inputLayer, maskLayer in

            let maskSum = MaskSumLayer(
                graph: graph,
                maskTensor: maskLayer.tensor)

            let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(
                graph: graph,
                maskSum: maskSum)

            let block =
                GlobalPoolingResidualBlock(
                    graph: graph,
                    sourceTensor: inputLayer.tensor,
                    maskTensor: maskLayer.tensor,
                    maskSumTensor: maskSum.tensor,
                    maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
                    descriptor: descriptor,
                    nnXLen: nnXLen,
                    nnYLen: nnYLen)

            return block.resultTensor
        }
    }

    /// Initialize a GlobalPoolingResidualBlock object
    ///
    /// - Parameters:
    ///   - graph: The graph
    ///   - sourceTensor: The source tensor
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The mask sum tensor
    ///   - maskSumSqrtS14M01Tensor: The mask sum square tensor
    ///   - descriptor: The descriptor of the global pooling residual block
    ///   - nnXLen: The X length
    ///   - nnYLen: The Y length
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        descriptor: SWGlobalPoolingResidualBlockDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {
        let maskSum = MaskSumLayer(tensor: maskSumTensor)
        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(tensor: maskSumSqrtS14M01Tensor)

        let preBN = BatchNormLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.preBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let preActivation = ActivationLayer(
            graph: graph,
            sourceTensor: preBN.resultTensor,
            activationKind: descriptor.preActivation)

        let regularConv = ConvLayer(
            graph: graph,
            sourceTensor: preActivation.resultTensor,
            descriptor: descriptor.regularConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let gpoolConv = ConvLayer(
            graph: graph,
            sourceTensor: preActivation.resultTensor,
            descriptor: descriptor.gpoolConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let gpoolBN = BatchNormLayer(
            graph: graph,
            sourceTensor: gpoolConv.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.gpoolBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let gpoolActivation = ActivationLayer(
            graph: graph,
            sourceTensor: gpoolBN.resultTensor,
            activationKind: descriptor.gpoolActivation)

        let gpoolConcat = GlobalPoolingLayer(
            graph: graph,
            sourceTensor: gpoolActivation.resultTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor)

        assert(gpoolConcat.resultTensor.shape?[1] == descriptor.gpoolToBiasMul.inChannels)

        let gpoolToBiasMul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.gpoolToBiasMul,
            sourceTensor: gpoolConcat.resultTensor)

        let added = AddNCBiasLayer(
            graph: graph,
            sourceTensor: regularConv.resultTensor,
            biasTensor: gpoolToBiasMul.resultTensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.gpoolToBiasMul.outChannels)

        let midBN = BatchNormLayer(
            graph: graph,
            sourceTensor: added.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.midBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let midActivation = ActivationLayer(
            graph: graph,
            sourceTensor: midBN.resultTensor,
            activationKind: descriptor.midActivation)

        let finalConv = ConvLayer(
            graph: graph,
            sourceTensor: midActivation.resultTensor,
            descriptor: descriptor.finalConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        resultTensor = graph.addition(
            sourceTensor,
            finalConv.resultTensor,
            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

public func testGlobalPoolingResidualBlock(
    descriptor: SWGlobalPoolingResidualBlockDesc,
    batchSize: Int32,
    nnXLen: Int32,
    nnYLen: Int32,
    input: UnsafeMutablePointer<Float32>,
    mask: UnsafeMutablePointer<Float32>,
    output: UnsafeMutablePointer<Float32>
) {
    GlobalPoolingResidualBlock.test(
        descriptor: descriptor,
        batchSize: batchSize as NSNumber,
        nnXLen: nnXLen as NSNumber,
        nnYLen: nnYLen as NSNumber,
        input: input,
        mask: mask,
        output: output)
}

/// A class that represents a nested bottleneck residual block
public class SWNestedBottleneckResidualBlockDesc: BlockDescriptor {
    /// The batch normalization layer before the residual block.
    let preBN: SWBatchNormLayerDesc

    /// The pre-activation function of the residual block.
    let preActivation: ActivationKind

    /// The convolutional layer before the residual block.
    let preConv: SWConvLayerDesc

    /// The list of blocks that make up the trunk
    let blockDescriptors: [BlockDescriptor]

    /// The batch normalization layer after the residual block.
    let postBN: SWBatchNormLayerDesc

    /// The activation function after the post batch normalization layer.
    let postActivation: ActivationKind

    /// The convolutional layer after the post activation layer.
    let postConv: SWConvLayerDesc

    /// Initialize a SWNestedBottleneckResidualBlockDesc object.
    /// - Parameters:
    ///   - preBN: The batch normalization layer before the residual block.
    ///   - preActivation: The pre-activation function of the residual block.
    ///   - preConv: The convolutional layer before the residual block.
    ///   - postBN: The batch normalization layer after the residual block.
    ///   - postActivation: The activation function after the post batch normalization layer.
    ///   - postConv: The convolutional layer after the post activation layer.
    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        preConv: SWConvLayerDesc,
        blockDescriptors: [BlockDescriptor],
        postBN: SWBatchNormLayerDesc,
        postActivation: ActivationKind,
        postConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.preConv = preConv
        self.blockDescriptors = blockDescriptors
        self.postBN = postBN
        self.postActivation = postActivation
        self.postConv = postConv
    }
}

public func createSWNestedBottleneckResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    preConv: SWConvLayerDesc,
    blockDescriptors: [BlockDescriptor],
    postBN: SWBatchNormLayerDesc,
    postActivation: ActivationKind,
    postConv: SWConvLayerDesc
) -> SWNestedBottleneckResidualBlockDesc {
    return SWNestedBottleneckResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        preConv: preConv,
        blockDescriptors: blockDescriptors,
        postBN: postBN,
        postActivation: postActivation,
        postConv: postConv)
}

public class BlockDescriptor {
}

public class BlockDescriptorBuilder {
    public var blockDescriptors: [BlockDescriptor] = []

    public func enque(with descriptor: BlockDescriptor) {
        blockDescriptors.append(descriptor)
    }
}

public func createBlockDescriptorBuilder() -> BlockDescriptorBuilder {
    return BlockDescriptorBuilder()
}

/// A structure that represents a block stack
struct BlockStack {
    /// The resulting tensor after processing the block stack
    let resultTensor: MPSGraphTensor

    /// Process block descriptors
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - blockDescriptors: The block descriptors
    ///   - index: The index of the block descriptor
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    /// - Returns: The result tensor
    static func processBlockDescriptors(
        _ graph: MPSGraph,
        _ sourceTensor: MPSGraphTensor,
        _ maskTensor: MPSGraphTensor,
        _ maskSumTensor: MPSGraphTensor,
        _ maskSumSqrtS14M01Tensor: MPSGraphTensor,
        _ blockDescriptors: [BlockDescriptor],
        _ index: Int,
        _ nnXLen: NSNumber,
        _ nnYLen: NSNumber
    ) -> MPSGraphTensor {
        guard index < blockDescriptors.count else {
            return sourceTensor
        }

        let blockDescriptor = blockDescriptors[index]
        let blockInput: MPSGraphTensor

        switch blockDescriptor {
        case let globalPoolingDescriptor as SWGlobalPoolingResidualBlockDesc:
            let globalPooling = GlobalPoolingResidualBlock(
                graph: graph,
                sourceTensor: sourceTensor,
                maskTensor: maskTensor,
                maskSumTensor: maskSumTensor,
                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                descriptor: globalPoolingDescriptor,
                nnXLen: nnXLen,
                nnYLen: nnYLen)

            blockInput = globalPooling.resultTensor
        case let nestedBottleneckDescriptor as SWNestedBottleneckResidualBlockDesc:
            let nestedBottleneck = NestedBottleneckResidualBlock(
                graph: graph,
                sourceTensor: sourceTensor,
                maskTensor: maskTensor,
                maskSumTensor: maskSumTensor,
                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                descriptor: nestedBottleneckDescriptor,
                nnXLen: nnXLen,
                nnYLen: nnYLen)

            blockInput = nestedBottleneck.resultTensor
        case let residualBlockDescriptor as SWResidualBlockDesc:
            let ordinary = ResidualBlock(
                graph: graph,
                sourceTensor: sourceTensor,
                maskTensor: maskTensor,
                descriptor: residualBlockDescriptor,
                nnXLen: nnXLen,
                nnYLen: nnYLen)

            blockInput = ordinary.resultTensor
        default:
            blockInput = sourceTensor
        }

        return processBlockDescriptors(
            graph,
            blockInput,
            maskTensor,
            maskSumTensor,
            maskSumSqrtS14M01Tensor,
            blockDescriptors,
            index + 1,
            nnXLen,
            nnYLen)
    }

    /// Initialize a BlockStack object
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - blockDescriptors: The block descriptors
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        blockDescriptors: [BlockDescriptor],
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {
        resultTensor = BlockStack.processBlockDescriptors(
            graph,
            sourceTensor,
            maskTensor,
            maskSumTensor,
            maskSumSqrtS14M01Tensor,
            blockDescriptors,
            0,
            nnXLen,
            nnYLen)
    }
}

/// A structure that represents a nested bottleneck residual block
struct NestedBottleneckResidualBlock {
    /// The resulting tensor after processing the nested bottleneck residual block
    let resultTensor: MPSGraphTensor

    /// Initialize a ResidualBlock object
    ///
    /// - Parameters:
    ///   - graph: The MPSGraph
    ///   - sourceTensor: The input tensor
    ///   - maskTensor: The mask tensor
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - descriptor: The nested bottleneck residual block descriptor
    ///   - nnXLen: X length
    ///   - nnYLen: Y length
    init(
        graph: MPSGraph,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        descriptor: SWNestedBottleneckResidualBlockDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {

        let preBN = BatchNormLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.preBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let preActivation = ActivationLayer(
            graph: graph,
            sourceTensor: preBN.resultTensor,
            activationKind: descriptor.preActivation)

        let preConv = ConvLayer(
            graph: graph,
            sourceTensor: preActivation.resultTensor,
            descriptor: descriptor.preConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let blocks = BlockStack(
            graph: graph,
            sourceTensor: preConv.resultTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSumTensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
            blockDescriptors: descriptor.blockDescriptors,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let postBN = BatchNormLayer(
            graph: graph,
            sourceTensor: blocks.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.postBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let postActivation = ActivationLayer(
            graph: graph,
            sourceTensor: postBN.resultTensor,
            activationKind: descriptor.postActivation)

        let postConv = ConvLayer(
            graph: graph,
            sourceTensor: postActivation.resultTensor,
            descriptor: descriptor.postConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        resultTensor = graph.addition(
            sourceTensor,
            postConv.resultTensor,
            name: nil)

        assert(resultTensor.shape?.count == 4)
    }
}

/// Class representing the description of the SGF Metadata Encoder.
///
/// This encoder consists of three matrix multiplication layers, each followed by a bias and an activation function.
public class SWSGFMetadataEncoderDesc {
    /// Version of the SGF Metadata Encoder.
    let version: Int

    /// Number of input metadata channels.
    let numInputMetaChannels: Int

    /// Description of the first multiplication layer.
    let mul1: SWMatMulLayerDesc

    /// Description of the bias for the first layer.
    let bias1: SWMatBiasLayerDesc

    /// Activation kind for the first layer.
    let act1: ActivationKind

    /// Description of the second multiplication layer.
    let mul2: SWMatMulLayerDesc

    /// Description of the bias for the second layer.
    let bias2: SWMatBiasLayerDesc

    /// Activation kind for the second layer.
    let act2: ActivationKind

    /// Description of the third multiplication layer.
    let mul3: SWMatMulLayerDesc

    /// Initializes a new instance of the `SWSGFMetadataEncoderDesc` class.
    ///
    /// - Parameters:
    ///   - version: The version of the SGF Metadata Encoder.
    ///   - numInputMetaChannels: The number of input metadata channels.
    ///   - mul1: Description of the first multiplication layer.
    ///   - bias1: Description of the bias for the first layer.
    ///   - act1: Activation kind for the first layer.
    ///   - mul2: Description of the second multiplication layer.
    ///   - bias2: Description of the bias for the second layer.
    ///   - act2: Activation kind for the second layer.
    ///   - mul3: Description of the third multiplication layer.
    init(
        version: Int,
        numInputMetaChannels: Int,
        mul1: SWMatMulLayerDesc,
        bias1: SWMatBiasLayerDesc,
        act1: ActivationKind,
        mul2: SWMatMulLayerDesc,
        bias2: SWMatBiasLayerDesc,
        act2: ActivationKind,
        mul3: SWMatMulLayerDesc
    ) {
        self.version = version
        self.numInputMetaChannels = numInputMetaChannels
        self.mul1 = mul1
        self.bias1 = bias1
        self.act1 = act1
        self.mul2 = mul2
        self.bias2 = bias2
        self.act2 = act2
        self.mul3 = mul3
    }
}

/// Creates an instance of `SWSGFMetadataEncoderDesc` using the specified parameters.
///
/// - Parameters:
///   - version: An `Int32` representing the version of the encoder descriptor.
///   - numInputMetaChannels: An `Int32` specifying the number of input metadata channels.
///   - mul1: A `SWMatMulLayerDesc` representing the description of the first matrix multiplication layer.
///   - bias1: A `SWMatBiasLayerDesc` representing the description of the bias for the first layer.
///   - act1: An `ActivationKind` specifying the activation function applied after the first layer.
///   - mul2: A `SWMatMulLayerDesc` representing the description of the second matrix multiplication layer.
///   - bias2: A `SWMatBiasLayerDesc` representing the description of the bias for the second layer.
///   - act2: An `ActivationKind` specifying the activation function applied after the second layer.
///   - mul3: A `SWMatMulLayerDesc` representing the description of the third matrix multiplication layer.
///
/// - Returns:
///   An instance of `SWSGFMetadataEncoderDesc` initialized with the provided parameters.
public func createSWSGFMetadataEncoderDesc(
    version: Int32,
    numInputMetaChannels: Int32,
    mul1: SWMatMulLayerDesc,
    bias1: SWMatBiasLayerDesc,
    act1: ActivationKind,
    mul2: SWMatMulLayerDesc,
    bias2: SWMatBiasLayerDesc,
    act2: ActivationKind,
    mul3: SWMatMulLayerDesc
) -> SWSGFMetadataEncoderDesc? {
    return SWSGFMetadataEncoderDesc(
        version: Int(version),
        numInputMetaChannels: Int(numInputMetaChannels),
        mul1: mul1,
        bias1: bias1,
        act1: act1,
        mul2: mul2,
        bias2: bias2,
        act2: act2,
        mul3: mul3)
}

/// A class that describes SGF metadata encoder.
/// SGFMetadataEncoder takes a graph, a descriptor object defining various parameters for the encoding process,
/// and an input tensor, and performs a sequence of matrix multiplications, bias additions, and activation functions
/// to produce a final encoded tensor.
class SGFMetadataEncoder {
    /// The resulting tensor after encoding the metadata.
    let resultTensor: MPSGraphTensor

    /// Initializes an `SGFMetadataEncoder` instance and performs the encoding process.
    ///
    /// - Parameters:
    ///   - graph: The computational graph object used to define and manage tensor operations.
    ///   - descriptor: An object holding all the required parameters, including matrix multiplication, biases,
    ///                 and activation functions for each layer.
    ///   - sourceTensor: The initial input tensor containing the metadata to be encoded.
    init(
        graph: MPSGraph,
        descriptor: SWSGFMetadataEncoderDesc,
        sourceTensor: MPSGraphTensor
    ) {

        // First matrix multiplication layer.
        let mul1 = MatMulLayer(
            graph: graph,
            descriptor: descriptor.mul1,
            sourceTensor: sourceTensor)

        // Adding bias to the result of the first matrix multiplication.
        let bias1 = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.bias1,
            sourceTensor: mul1.resultTensor)

        // Applying the first activation function to the biased tensor.
        let act1 = ActivationLayer(
            graph: graph,
            sourceTensor: bias1.resultTensor,
            activationKind: descriptor.act1)

        // Second matrix multiplication layer taking the output of the first activation layer.
        let mul2 = MatMulLayer(
            graph: graph,
            descriptor: descriptor.mul2,
            sourceTensor: act1.resultTensor)

        // Adding bias to the result of the second matrix multiplication.
        let bias2 = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.bias2,
            sourceTensor: mul2.resultTensor)

        // Applying the second activation function to the biased tensor.
        let act2 = ActivationLayer(
            graph: graph,
            sourceTensor: bias2.resultTensor,
            activationKind: descriptor.act2)

        // Third and final matrix multiplication layer taking the output of the second activation layer.
        let mul3 = MatMulLayer(
            graph: graph,
            descriptor: descriptor.mul3,
            sourceTensor: act2.resultTensor)

        // Setting the final result tensor to the output of the last matrix multiplication layer.
        resultTensor = mul3.resultTensor

        assert(resultTensor.shape?.count == 2)
    }
}

/// A class that describes a trunk for a neural network
public class SWTrunkDesc {
    /// The version of the ResNet trunk
    let version: Int
    /// Number of channels for the trunk
    let trunkNumChannels: NSNumber
    /// Number of channels for the mid section
    let midNumChannels: NSNumber
    /// Number of channels for the regular section
    let regularNumChannels: NSNumber
    /// Number of channels for the global pooling section
    let gpoolNumChannels: NSNumber
    /// The description of the initial convolutional layer
    let initialConv: SWConvLayerDesc
    /// The description of the initial matrix multiplication layer
    let initialMatMul: SWMatMulLayerDesc
    /// The description of the SGF metadata encoder
    let sgfMetadataEncoder: SWSGFMetadataEncoderDesc?
    /// The list of blocks that make up the trunk
    let blockDescriptors: [BlockDescriptor]
    /// The description of the batch normalization layer that is applied at the end of the trunk
    let trunkTipBN: SWBatchNormLayerDesc
    /// The activation function that is applied at the end of the trunk
    let trunkTipActivation: ActivationKind

    /// Initializes a SWTrunkDesc object
    /// - Parameters:
    ///   - version: The version of the ResNet trunk
    ///   - trunkNumChannels: Number of channels for the trunk
    ///   - midNumChannels: Number of channels for the mid section
    ///   - regularNumChannels: Number of channels for the regular section
    ///   - gpoolNumChannels: Number of channels for the global pooling section
    ///   - initialConv: The description of the initial convolutional layer
    ///   - initialMatMul: The description of the initial matrix multiplication layer
    ///   - sgfMetadataEncoder: The description of the SGF metadata encoder
    ///   - blockDescriptors: The list of blocks that make up the trunk
    ///   - trunkTipBN: The description of the batch normalization layer that is applied at the end of the trunk
    ///   - trunkTipActivation: The activation function that is applied at the end of the trunk
    init(
        version: Int,
        trunkNumChannels: NSNumber,
        midNumChannels: NSNumber,
        regularNumChannels: NSNumber,
        gpoolNumChannels: NSNumber,
        initialConv: SWConvLayerDesc,
        initialMatMul: SWMatMulLayerDesc,
        sgfMetadataEncoder: SWSGFMetadataEncoderDesc?,
        blockDescriptors: [BlockDescriptor],
        trunkTipBN: SWBatchNormLayerDesc,
        trunkTipActivation: ActivationKind
    ) {
        self.version = version
        self.trunkNumChannels = trunkNumChannels
        self.midNumChannels = midNumChannels
        self.regularNumChannels = regularNumChannels
        self.gpoolNumChannels = gpoolNumChannels
        self.initialConv = initialConv
        self.initialMatMul = initialMatMul
        self.sgfMetadataEncoder = sgfMetadataEncoder
        self.blockDescriptors = blockDescriptors
        self.trunkTipBN = trunkTipBN
        self.trunkTipActivation = trunkTipActivation
    }
}

public func createSWTrunkDesc(
    version: Int32,
    trunkNumChannels: Int32,
    midNumChannels: Int32,
    regularNumChannels: Int32,
    gpoolNumChannels: Int32,
    initialConv: SWConvLayerDesc,
    initialMatMul: SWMatMulLayerDesc,
    sgfMetadataEncoder: SWSGFMetadataEncoderDesc?,
    blockDescriptors: [BlockDescriptor],
    trunkTipBN: SWBatchNormLayerDesc,
    trunkTipActivation: ActivationKind
) -> SWTrunkDesc {
    return SWTrunkDesc(
        version: Int(version),
        trunkNumChannels: trunkNumChannels as NSNumber,
        midNumChannels: midNumChannels as NSNumber,
        regularNumChannels: regularNumChannels as NSNumber,
        gpoolNumChannels: gpoolNumChannels as NSNumber,
        initialConv: initialConv,
        initialMatMul: initialMatMul,
        sgfMetadataEncoder: sgfMetadataEncoder,
        blockDescriptors: blockDescriptors,
        trunkTipBN: trunkTipBN,
        trunkTipActivation: trunkTipActivation)
}

/// A structure representing a ResNet trunk for a neural network
struct Trunk {
    /// The resulting tensor after processing the trunk
    let resultTensor: MPSGraphTensor

    /// Returns the block source tensor by processing the input meta tensor, if available, and adding a bias term.
    ///
    /// - Parameters:
    ///     - graph: The Metal Performance Shaders (MPS) graph.
    ///     - descriptor: The SGF metadata encoder descriptor.
    ///     - initialAdd: The initial add operation result tensor.
    ///     - inputMetaTensor: The input meta tensor.
    ///     - nnXLen: The X length of the neural network (NN).
    ///     - nnYLen: The Y length of the neural network (NN).
    ///     - numChannels: The number of channels of the initial add operation result tensor.
    ///
    /// - Returns:
    ///     - blockSourceTensor: The processed block source tensor.
    ///
    /// This function is used to get the block source tensor by processing the input meta tensor, if available.
    /// If the input meta tensor is not available, it returns the result tensor from the initial add operation.
    /// The function uses SGF metadata encoder and AddNCBiasLayer to process the input meta tensor.
    static func getBlockSourceTensor(
        graph: MPSGraph,
        descriptor: SWSGFMetadataEncoderDesc?,
        initialAdd: AddNCBiasLayer,
        inputMetaTensor: MPSGraphTensor?,
        nnXLen: NSNumber,
        nnYLen: NSNumber,
        numChannels: NSNumber
    ) -> MPSGraphTensor {
        var blockSourceTensor: MPSGraphTensor

        if let inputMetaTensor,
            let descriptor, descriptor.numInputMetaChannels > 0
        {
            let encoded = SGFMetadataEncoder(
                graph: graph,
                descriptor: descriptor,
                sourceTensor: inputMetaTensor)

            let encodedAdd = AddNCBiasLayer(
                graph: graph,
                sourceTensor: initialAdd.resultTensor,
                biasTensor: encoded.resultTensor,
                nnXLen: nnXLen,
                nnYLen: nnYLen,
                numChannels: numChannels)

            blockSourceTensor = encodedAdd.resultTensor
        } else {
            blockSourceTensor = initialAdd.resultTensor
        }

        return blockSourceTensor
    }

    /// Initializes a Trunk object
    /// - Parameters:
    ///   - graph: The graph used to build the trunk
    ///   - descriptor: A SWTrunkDesc object that describes the trunk
    ///   - inputTensor: The input tensor
    ///   - inputGlobalTensor: The input global tensor
    ///   - inputMetaTensor: The input meta tensor
    ///   - maskTensor: The tensor used to mask input activations
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor
    ///   - nnXLen: The length of the X dimension of the input tensor
    ///   - nnYLen: The length of the Y dimension of the input tensor
    init(
        graph: MPSGraph,
        descriptor: SWTrunkDesc,
        inputTensor: MPSGraphTensor,
        inputGlobalTensor: MPSGraphTensor,
        inputMetaTensor: MPSGraphTensor?,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {

        let initialConv = ConvLayer(
            graph: graph,
            sourceTensor: inputTensor,
            descriptor: descriptor.initialConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let initialMatMul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.initialMatMul,
            sourceTensor: inputGlobalTensor)

        let initialAdd = AddNCBiasLayer(
            graph: graph,
            sourceTensor: initialConv.resultTensor,
            biasTensor: initialMatMul.resultTensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.initialMatMul.outChannels)

        let blockSourceTensor = Trunk.getBlockSourceTensor(
            graph: graph,
            descriptor: descriptor.sgfMetadataEncoder,
            initialAdd: initialAdd,
            inputMetaTensor: inputMetaTensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.initialMatMul.outChannels)

        let blocks = BlockStack(
            graph: graph,
            sourceTensor: blockSourceTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSumTensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
            blockDescriptors: descriptor.blockDescriptors,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let trunkTipBN = BatchNormLayer(
            graph: graph,
            sourceTensor: blocks.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.trunkTipBN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let trunkTipActivation = ActivationLayer(
            graph: graph,
            sourceTensor: trunkTipBN.resultTensor,
            activationKind: descriptor.trunkTipActivation)

        resultTensor = trunkTipActivation.resultTensor

        assert(resultTensor.shape?.count == 4)
    }
}

/// A class that describes a policy head for a neural network, responsible for predicting
/// the best moves for the current player and the opposing player on the subsequent turn.
public struct SWPolicyHeadDesc {
    /// The version of the policy head
    let version: Int
    /// The 1x1 convolution layer for P
    let p1Conv: SWConvLayerDesc
    /// The 1x1 convolution layer for G
    let g1Conv: SWConvLayerDesc
    /// The batch normalization layer for G
    let g1BN: SWBatchNormLayerDesc
    /// The activation function for G
    let g1Activation: ActivationKind
    /// The global pooling bias structure that pools the output of G to bias the output of P
    let gpoolToBiasMul: SWMatMulLayerDesc
    /// The batch normalization layer for P
    let p1BN: SWBatchNormLayerDesc
    /// The activation function for P
    let p1Activation: ActivationKind
    /// The 1x1 convolution layer with 2 channels for outputting two policy distributions
    let p2Conv: SWConvLayerDesc
    /// The fully connected linear layer for outputting logits for the pass move
    let gpoolToPassMul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer for model version >= 15
    let gpoolToPassBias: SWMatBiasLayerDesc?
    /// The activation function for the bias layer in model version >= 15
    let passActivation: ActivationKind?
    /// The fully connected linear layer for outputting logits for the pass move in model version >= 15
    let gpoolToPassMul2: SWMatMulLayerDesc?

    /// Initializes a SWPolicyHeadDesc object with the given parameters
    /// - Parameters:
    ///   - version: The version of the policy head
    ///   - p1Conv: The 1x1 convolution layer for P
    ///   - g1Conv: The 1x1 convolution layer for G
    ///   - g1BN: The batch normalization layer for G
    ///   - g1Activation: The activation function for G
    ///   - gpoolToBiasMul: The global pooling bias structure that pools the output of G to bias the output of P
    ///   - p1BN: The batch normalization layer for P
    ///   - p1Activation: The activation function for P
    ///   - p2Conv: The 1x1 convolution layer with 2 channels for outputting two policy distributions
    ///   - gpoolToPassMul: The fully connected linear layer for outputting logits for the pass move
    init(
        version: Int,
        p1Conv: SWConvLayerDesc,
        g1Conv: SWConvLayerDesc,
        g1BN: SWBatchNormLayerDesc,
        g1Activation: ActivationKind,
        gpoolToBiasMul: SWMatMulLayerDesc,
        p1BN: SWBatchNormLayerDesc,
        p1Activation: ActivationKind,
        p2Conv: SWConvLayerDesc,
        gpoolToPassMul: SWMatMulLayerDesc,
        gpoolToPassBias: SWMatBiasLayerDesc?,
        passActivation: ActivationKind?,
        gpoolToPassMul2: SWMatMulLayerDesc?
    ) {
        self.version = version
        self.p1Conv = p1Conv
        self.g1Conv = g1Conv
        self.g1BN = g1BN
        self.g1Activation = g1Activation
        self.gpoolToBiasMul = gpoolToBiasMul
        self.p1BN = p1BN
        self.p1Activation = p1Activation
        self.p2Conv = p2Conv
        self.gpoolToPassMul = gpoolToPassMul
        self.gpoolToPassBias = gpoolToPassBias
        self.passActivation = passActivation
        self.gpoolToPassMul2 = gpoolToPassMul2

        assert(
            (version >= 15)
                || ((gpoolToPassBias == nil) && (passActivation == nil) && (gpoolToPassMul2 == nil))
        )
        assert(
            (version < 15)
                || ((gpoolToPassBias != nil) && (passActivation != nil) && (gpoolToPassMul2 != nil))
        )
    }
}

public func createSWPolicyHeadDesc(
    version: Int32,
    p1Conv: SWConvLayerDesc,
    g1Conv: SWConvLayerDesc,
    g1BN: SWBatchNormLayerDesc,
    g1Activation: ActivationKind,
    gpoolToBiasMul: SWMatMulLayerDesc,
    p1BN: SWBatchNormLayerDesc,
    p1Activation: ActivationKind,
    p2Conv: SWConvLayerDesc,
    gpoolToPassMul: SWMatMulLayerDesc,
    gpoolToPassBias: SWMatBiasLayerDesc,
    passActivation: ActivationKind,
    gpoolToPassMul2: SWMatMulLayerDesc
) -> SWPolicyHeadDesc {
    if version >= 15 {
        return SWPolicyHeadDesc(
            version: Int(version),
            p1Conv: p1Conv,
            g1Conv: g1Conv,
            g1BN: g1BN,
            g1Activation: g1Activation,
            gpoolToBiasMul: gpoolToBiasMul,
            p1BN: p1BN,
            p1Activation: p1Activation,
            p2Conv: p2Conv,
            gpoolToPassMul: gpoolToPassMul,
            gpoolToPassBias: gpoolToPassBias,
            passActivation: passActivation,
            gpoolToPassMul2: gpoolToPassMul2)
    } else {
        return SWPolicyHeadDesc(
            version: Int(version),
            p1Conv: p1Conv,
            g1Conv: g1Conv,
            g1BN: g1BN,
            g1Activation: g1Activation,
            gpoolToBiasMul: gpoolToBiasMul,
            p1BN: p1BN,
            p1Activation: p1Activation,
            p2Conv: p2Conv,
            gpoolToPassMul: gpoolToPassMul,
            gpoolToPassBias: nil,
            passActivation: nil,
            gpoolToPassMul2: nil)
    }
}

/// A structure that represents a policy head of a neural network.
struct PolicyHead {
    /// The tensor that holds the policy prediction of the neural network
    let policyTensor: MPSGraphTensor
    /// The tensor that holds the policy pass of the neural network
    let policyPassTensor: MPSGraphTensor

    /// Initializes a PolicyHead object
    /// - Parameters:
    ///   - graph: The MPSGraph object to which the policy head is added
    ///   - descriptor: The description of the policy head
    ///   - sourceTensor: The input tensor to the policy head
    ///   - maskTensor: The mask tensor for the input tensor
    ///   - maskSumTensor: The sum of the mask tensor
    ///   - maskSumSqrtS14M01Tensor: The square root of the sum of the mask tensor and a small epsilon
    ///   - nnXLen: The number of X pixels in the input tensor
    ///   - nnYLen: The number of Y pixels in the input tensor
    init(
        graph: MPSGraph,
        descriptor: SWPolicyHeadDesc,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {

        let p1Conv = ConvLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            descriptor: descriptor.p1Conv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let g1Conv = ConvLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            descriptor: descriptor.g1Conv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let g1BN = BatchNormLayer(
            graph: graph,
            sourceTensor: g1Conv.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.g1BN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let g1Activation = ActivationLayer(
            graph: graph,
            sourceTensor: g1BN.resultTensor,
            activationKind: descriptor.g1Activation)

        let g1Concat = GlobalPoolingLayer(
            graph: graph,
            sourceTensor: g1Activation.resultTensor,
            maskTensor: maskTensor,
            maskSumTensor: maskSumTensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor)

        assert(g1Concat.resultTensor.shape?[1] == descriptor.gpoolToBiasMul.inChannels)

        let gpoolToBiasMul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.gpoolToBiasMul,
            sourceTensor: g1Concat.resultTensor)

        let added = AddNCBiasLayer(
            graph: graph,
            sourceTensor: p1Conv.resultTensor,
            biasTensor: gpoolToBiasMul.resultTensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.gpoolToBiasMul.outChannels)

        let p1BN = BatchNormLayer(
            graph: graph,
            sourceTensor: added.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.p1BN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let p1Activation = ActivationLayer(
            graph: graph,
            sourceTensor: p1BN.resultTensor,
            activationKind: descriptor.p1Activation)

        let p2Conv = ConvLayer(
            graph: graph,
            sourceTensor: p1Activation.resultTensor,
            descriptor: descriptor.p2Conv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        policyTensor = p2Conv.resultTensor

        assert(g1Concat.resultTensor.shape?[1] == descriptor.gpoolToPassMul.inChannels)

        let gpoolToPassMul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.gpoolToPassMul,
            sourceTensor: g1Concat.resultTensor)

        if let gpoolToPassBias = descriptor.gpoolToPassBias,
            let passActivation = descriptor.passActivation,
            let gpoolToPassMul2 = descriptor.gpoolToPassMul2
        {
            assert(descriptor.version >= 15)

            let gpoolToPassBiasLayer = MatBiasLayer(
                graph: graph,
                descriptor: gpoolToPassBias,
                sourceTensor: gpoolToPassMul.resultTensor)

            let passActivationLayer = ActivationLayer(
                graph: graph,
                sourceTensor: gpoolToPassBiasLayer.resultTensor,
                activationKind: passActivation)

            let gpoolToPassMul2Layer = MatMulLayer(
                graph: graph,
                descriptor: gpoolToPassMul2,
                sourceTensor: passActivationLayer.resultTensor)

            policyPassTensor = gpoolToPassMul2Layer.resultTensor
        } else {
            assert(descriptor.version < 15)
            policyPassTensor = gpoolToPassMul.resultTensor
        }

        assert(policyTensor.shape?.count == 4)
        assert(policyPassTensor.shape?.count == 2)
    }
}

/// A struct that describes the value head of a neural network
public struct SWValueHeadDesc {
    /// The version of the value head
    let version: Int
    /// The description of the first convolutional layer in the value head
    let v1Conv: SWConvLayerDesc
    /// The description of the batch normalization layer after the first convolutional layer in the value head
    let v1BN: SWBatchNormLayerDesc
    /// The activation function that is applied after the first batch normalization layer in the value head
    let v1Activation: ActivationKind
    /// The description of the matrix multiplication layer that is applied to the output of the first convolutional layer in the value head
    let v2Mul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    let v2Bias: SWMatBiasLayerDesc
    /// The activation function that is applied after the bias layer in the value head
    let v2Activation: ActivationKind
    /// The description of the matrix multiplication layer that is applied to the output of the bias layer in the value head
    let v3Mul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    let v3Bias: SWMatBiasLayerDesc
    /// The description of the matrix multiplication layer that is applied to the output of the third bias layer in the value head
    let sv3Mul: SWMatMulLayerDesc
    /// The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    let sv3Bias: SWMatBiasLayerDesc
    /// The description of the convolutional layer that is applied to the board ownership map in the value head
    let vOwnershipConv: SWConvLayerDesc

    /// Initializes a SWValueHeadDesc object
    /// - Parameters:
    ///   - version: The version of the value head
    ///   - v1Conv: The description of the first convolutional layer in the value head
    ///   - v1BN: The description of the batch normalization layer after the first convolutional layer in the value head
    ///   - v1Activation: The activation function that is applied after the first batch normalization layer in the value head
    ///   - v2Mul: The description of the matrix multiplication layer that is applied to the output of the first convolutional layer in the value head
    ///   - v2Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - v2Activation: The activation function that is applied after the bias layer in the value head
    ///   - v3Mul: The description of the matrix multiplication layer that is applied to the output of the bias layer in the value head
    ///   - v3Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - sv3Mul: The description of the matrix multiplication layer that is applied to the output of the third bias layer in the value head
    ///   - sv3Bias: The description of the bias layer that is applied to the output of the matrix multiplication layer in the value head
    ///   - vOwnershipConv: The description of the convolutional layer that is applied to the board ownership map in the value head
    init(
        version: Int,
        v1Conv: SWConvLayerDesc,
        v1BN: SWBatchNormLayerDesc,
        v1Activation: ActivationKind,
        v2Mul: SWMatMulLayerDesc,
        v2Bias: SWMatBiasLayerDesc,
        v2Activation: ActivationKind,
        v3Mul: SWMatMulLayerDesc,
        v3Bias: SWMatBiasLayerDesc,
        sv3Mul: SWMatMulLayerDesc,
        sv3Bias: SWMatBiasLayerDesc,
        vOwnershipConv: SWConvLayerDesc
    ) {
        self.version = version
        self.v1Conv = v1Conv
        self.v1BN = v1BN
        self.v1Activation = v1Activation
        self.v2Mul = v2Mul
        self.v2Bias = v2Bias
        self.v2Activation = v2Activation
        self.v3Mul = v3Mul
        self.v3Bias = v3Bias
        self.sv3Mul = sv3Mul
        self.sv3Bias = sv3Bias
        self.vOwnershipConv = vOwnershipConv
    }
}

public func createSWValueHeadDesc(
    version: Int32,
    v1Conv: SWConvLayerDesc,
    v1BN: SWBatchNormLayerDesc,
    v1Activation: ActivationKind,
    v2Mul: SWMatMulLayerDesc,
    v2Bias: SWMatBiasLayerDesc,
    v2Activation: ActivationKind,
    v3Mul: SWMatMulLayerDesc,
    v3Bias: SWMatBiasLayerDesc,
    sv3Mul: SWMatMulLayerDesc,
    sv3Bias: SWMatBiasLayerDesc,
    vOwnershipConv: SWConvLayerDesc
) -> SWValueHeadDesc {
    return SWValueHeadDesc(
        version: Int(version),
        v1Conv: v1Conv,
        v1BN: v1BN,
        v1Activation: v1Activation,
        v2Mul: v2Mul,
        v2Bias: v2Bias,
        v2Activation: v2Activation,
        v3Mul: v3Mul,
        v3Bias: v3Bias,
        sv3Mul: sv3Mul,
        sv3Bias: sv3Bias,
        vOwnershipConv: vOwnershipConv)
}

/// A structure that creates a value head for the neural network, which produces the value, score value, and ownership tensors.
struct ValueHead {
    /// The tensor that represents the value of the board
    let valueTensor: MPSGraphTensor
    /// The tensor that represents the score value of the board
    let scoreValueTensor: MPSGraphTensor
    /// The tensor that represents the ownership of the board
    let ownershipTensor: MPSGraphTensor

    /// Initializes the value head using a graph, a descriptor, a source tensor, and other relevant tensors.
    /// - Parameters:
    ///   - graph: The graph used to perform calculations on tensors
    ///   - descriptor: The SWValueHeadDesc object that describes the value head
    ///   - sourceTensor: The tensor used to source data to the neural network
    ///   - maskTensor: The tensor used to mask out invalid moves
    ///   - maskSumTensor: The tensor used to sum up the mask tensor values
    ///   - maskSumSqrtS14M01Tensor: The tensor used to calculate a square root value
    ///   - maskSumSqrtS14M01SquareS01Tensor: The tensor used to calculate a square value
    ///   - nnXLen: The x-axis length of the neural network
    ///   - nnYLen: The y-axis length of the neural network
    init(
        graph: MPSGraph,
        descriptor: SWValueHeadDesc,
        sourceTensor: MPSGraphTensor,
        maskTensor: MPSGraphTensor,
        maskSumTensor: MPSGraphTensor,
        maskSumSqrtS14M01Tensor: MPSGraphTensor,
        maskSumSqrtS14M01SquareS01Tensor: MPSGraphTensor,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {

        let v1Conv = ConvLayer(
            graph: graph,
            sourceTensor: sourceTensor,
            descriptor: descriptor.v1Conv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let v1BN = BatchNormLayer(
            graph: graph,
            sourceTensor: v1Conv.resultTensor,
            maskTensor: maskTensor,
            descriptor: descriptor.v1BN,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let v1Activation = ActivationLayer(
            graph: graph,
            sourceTensor: v1BN.resultTensor,
            activationKind: descriptor.v1Activation)

        let v1Mean =
            GlobalPoolingValueLayer(
                graph: graph,
                sourceTensor: v1Activation.resultTensor,
                maskSumTensor: maskSumTensor,
                maskSumSqrtS14M01Tensor: maskSumSqrtS14M01Tensor,
                maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01Tensor)

        assert(v1Mean.resultTensor.shape?[1] == descriptor.v2Mul.inChannels)

        let v2Mul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.v2Mul,
            sourceTensor: v1Mean.resultTensor)

        let v2Bias = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.v2Bias,
            sourceTensor: v2Mul.resultTensor)

        let v2Activation = ActivationLayer(
            graph: graph,
            sourceTensor: v2Bias.resultTensor,
            activationKind: descriptor.v2Activation)

        let v3Mul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.v3Mul,
            sourceTensor: v2Activation.resultTensor)

        let v3Bias = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.v3Bias,
            sourceTensor: v3Mul.resultTensor)

        let sv3Mul = MatMulLayer(
            graph: graph,
            descriptor: descriptor.sv3Mul,
            sourceTensor: v2Activation.resultTensor)

        let sv3Bias = MatBiasLayer(
            graph: graph,
            descriptor: descriptor.sv3Bias,
            sourceTensor: sv3Mul.resultTensor)

        let vOwnershipConv = ConvLayer(
            graph: graph,
            sourceTensor: v1Activation.resultTensor,
            descriptor: descriptor.vOwnershipConv,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        valueTensor = v3Bias.resultTensor
        scoreValueTensor = sv3Bias.resultTensor
        ownershipTensor = vOwnershipConv.resultTensor

        assert(valueTensor.shape?.count == 2)
        assert(scoreValueTensor.shape?.count == 2)
        assert(ownershipTensor.shape?.count == 4)
    }
}

/// A struct that describes a neural network model used for playing the game of Go.
public struct SWModelDesc {
    /// The version of the model.
    let version: Int
    /// The name of the model.
    let name: String
    /// Number of channels for input features.
    let numInputChannels: NSNumber
    /// Number of channels for global input features.
    let numInputGlobalChannels: NSNumber
    /// Number of channels for meta input features.
    let numInputMetaChannels: NSNumber
    /// Number of channels for the value head output.
    let numValueChannels: NSNumber
    /// Number of channels for the score value head output.
    let numScoreValueChannels: NSNumber
    /// Number of channels for the ownership head output.
    let numOwnershipChannels: NSNumber
    /// The description of the trunk that makes up the backbone of the model.
    let trunk: SWTrunkDesc
    /// The description of the policy head that predicts the probability of playing at a particular position.
    let policyHead: SWPolicyHeadDesc
    /// The description of the value head that predicts the expected outcome of a game state.
    let valueHead: SWValueHeadDesc

    /// Initializes an SWModelDesc object.
    /// - Parameters:
    ///   - version: The version of the model.
    ///   - name: The name of the model.
    ///   - numInputChannels: Number of channels for input features.
    ///   - numInputGlobalChannels: Number of channels for global input features.
    ///   - numInputMetaChannels: Number of channels for meta input features.
    ///   - numValueChannels: Number of channels for the value head output.
    ///   - numScoreValueChannels: Number of channels for the score value head output.
    ///   - numOwnershipChannels: Number of channels for the ownership head output.
    ///   - trunk: The description of the trunk that makes up the backbone of the model.
    ///   - policyHead: The description of the policy head that predicts the probability of playing at a particular position.
    ///   - valueHead: The description of the value head that predicts the expected outcome of a game state.
    init(
        version: Int,
        name: String,
        numInputChannels: NSNumber,
        numInputGlobalChannels: NSNumber,
        numInputMetaChannels: NSNumber,
        numValueChannels: NSNumber,
        numScoreValueChannels: NSNumber,
        numOwnershipChannels: NSNumber,
        trunk: SWTrunkDesc,
        policyHead: SWPolicyHeadDesc,
        valueHead: SWValueHeadDesc
    ) {
        self.version = version
        self.name = name
        self.numInputChannels = numInputChannels
        self.numInputGlobalChannels = numInputGlobalChannels
        self.numInputMetaChannels = numInputMetaChannels
        self.numValueChannels = numValueChannels
        self.numScoreValueChannels = numScoreValueChannels
        self.numOwnershipChannels = numOwnershipChannels
        self.trunk = trunk
        self.policyHead = policyHead
        self.valueHead = valueHead
    }
}

public func createSWModelDesc(
    version: Int32,
    name: String,
    numInputChannels: Int32,
    numInputGlobalChannels: Int32,
    numInputMetaChannels: Int32,
    numValueChannels: Int32,
    numScoreValueChannels: Int32,
    numOwnershipChannels: Int32,
    trunk: SWTrunkDesc,
    policyHead: SWPolicyHeadDesc,
    valueHead: SWValueHeadDesc
) -> SWModelDesc {
    return SWModelDesc(
        version: Int(version),
        name: name,
        numInputChannels: numInputChannels as NSNumber,
        numInputGlobalChannels: numInputGlobalChannels as NSNumber,
        numInputMetaChannels: numInputMetaChannels as NSNumber,
        numValueChannels: numValueChannels as NSNumber,
        numScoreValueChannels: numScoreValueChannels as NSNumber,
        numOwnershipChannels: numOwnershipChannels as NSNumber,
        trunk: trunk,
        policyHead: policyHead,
        valueHead: valueHead)
}

/// A structure representing a neural network model for processing Go game states.
struct Model {
    /// The Metal device
    let device: MTLDevice
    /// The command queue used to execute the graph on the GPU
    let commandQueue: MTLCommandQueue
    /// The Metal Performance Shaders graph object used for building and executing the graph
    let graph: MPSGraph
    /// The length of the neural network input in the x dimension
    let nnXLen: NSNumber
    /// The length of the neural network input in the y dimension
    let nnYLen: NSNumber
    /// The version of the model
    let version: Int
    /// The number of channels in the value output layer
    let numValueChannels: NSNumber
    /// The number of channels in the score value output layer
    let numScoreValueChannels: NSNumber
    /// The number of channels in the ownership output layer
    let numOwnershipChannels: NSNumber
    /// The input layer of the neural network
    let input: InputLayer
    /// The global input layer of the neural network
    let inputGlobal: InputGlobalLayer
    /// The meta input layer of the neural network
    let inputMeta: InputMetaLayer
    /// The mask layer of the neural network
    let mask: MaskLayer
    /// The trunk of the neural network
    let trunk: Trunk
    /// The policy head of the neural network
    let policyHead: PolicyHead
    /// The value head of the neural network
    let valueHead: ValueHead
    /// The dictionary that maps the output tensors to the tensor data
    let targetTensors: [MPSGraphTensor]

    /// Initializes a Model object.
    /// - Parameters:
    ///   - device: The Metal device to use for computations.
    ///   - graph: The Metal Performance Shaders graph object used for building and executing the graph.
    ///   - descriptor: The description of the model.
    ///   - nnXLen: The length of the neural network input in the x dimension.
    ///   - nnYLen: The length of the neural network input in the y dimension.
    init(
        device: MTLDevice,
        graph: MPSGraph,
        descriptor: SWModelDesc,
        nnXLen: NSNumber,
        nnYLen: NSNumber
    ) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.graph = graph
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.version = descriptor.version
        self.numValueChannels = descriptor.numValueChannels
        self.numScoreValueChannels = descriptor.numScoreValueChannels
        self.numOwnershipChannels = descriptor.numOwnershipChannels

        input = InputLayer(
            graph: graph,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            numChannels: descriptor.numInputChannels)

        inputGlobal = InputGlobalLayer(
            graph: graph,
            numGlobalFeatures: descriptor.numInputGlobalChannels)

        inputMeta = InputMetaLayer(
            graph: graph,
            numMetaFeatures: descriptor.numInputMetaChannels)

        mask = MaskLayer(
            graph: graph,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        let maskSum = MaskSumLayer(
            graph: graph,
            maskTensor: mask.tensor)

        let maskSumSqrtS14M01 = MaskSumSqrtS14M01Layer(
            graph: graph,
            maskSum: maskSum)

        let maskSumSqrtS14M01SquareS01 = MaskSumSqrtS14M01SquareS01Layer(
            graph: graph,
            maskSumSqrtS14M01: maskSumSqrtS14M01)

        trunk = Trunk(
            graph: graph,
            descriptor: descriptor.trunk,
            inputTensor: input.tensor,
            inputGlobalTensor: inputGlobal.tensor,
            inputMetaTensor: inputMeta.tensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        policyHead = PolicyHead(
            graph: graph,
            descriptor: descriptor.policyHead,
            sourceTensor: trunk.resultTensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        valueHead = ValueHead(
            graph: graph,
            descriptor: descriptor.valueHead,
            sourceTensor: trunk.resultTensor,
            maskTensor: mask.tensor,
            maskSumTensor: maskSum.tensor,
            maskSumSqrtS14M01Tensor: maskSumSqrtS14M01.tensor,
            maskSumSqrtS14M01SquareS01Tensor: maskSumSqrtS14M01SquareS01.tensor,
            nnXLen: nnXLen,
            nnYLen: nnYLen)

        targetTensors = [
            policyHead.policyTensor,
            policyHead.policyPassTensor,
            valueHead.valueTensor,
            valueHead.scoreValueTensor,
            valueHead.ownershipTensor,
        ]
    }

    /// Applies the model to the given input data, and generates predictions for policy, value and ownership
    /// - Parameters:
    ///   - inputPointer: UnsafeMutablePointer to a flattened 2D array of floats representing the input state
    ///   - inputGlobalPointer: UnsafeMutablePointer to a flattened array of floats representing global state features
    ///   - inputMetaPointer: UnsafeMutablePointer to a flattened array of floats representing the metadata
    ///   - policy: UnsafeMutablePointer to a flattened 2D array of floats representing predicted policy
    ///   - policyPass: UnsafeMutablePointer to a flattened array of floats representing predicted probability of passing
    ///   - value: UnsafeMutablePointer to a flattened array of floats representing predicted value
    ///   - scoreValue: UnsafeMutablePointer to a flattened array of floats representing predicted score value
    ///   - ownership: UnsafeMutablePointer to a flattened 2D array of floats representing predicted ownership
    ///   - batchSize: The batch size
    func apply(
        input inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
        inputMeta inputMetaPointer: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {

        let channelAxis = InputShape.getChannelAxis()
        let numInputChannels = input.shape[channelAxis]

        let inputShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: numInputChannels,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        let inputDescriptor = MPSNDArrayDescriptor(
            dataType: input.tensor.dataType,
            shape: inputShape)

        let inputArray = MPSNDArray(
            device: device,
            descriptor: inputDescriptor)

        inputArray.writeBytes(inputPointer)

        let numInputGlobalChannels = inputGlobal.shape[channelAxis]

        let inputGlobalShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: numInputGlobalChannels,
            nnYLen: 1,
            nnXLen: 1)

        let inputGlobalDescriptor = MPSNDArrayDescriptor(
            dataType: inputGlobal.tensor.dataType,
            shape: inputGlobalShape)

        let inputGlobalArray = MPSNDArray(
            device: device,
            descriptor: inputGlobalDescriptor)

        inputGlobalArray.writeBytes(inputGlobalPointer)

        let numInputMetaChannels = inputMeta.shape[channelAxis]

        let inputMetaShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: numInputMetaChannels,
            nnYLen: 1,
            nnXLen: 1)

        let inputMetaDescriptor = MPSNDArrayDescriptor(
            dataType: inputMeta.tensor.dataType,
            shape: inputMetaShape)

        let inputMetaArray = MPSNDArray(
            device: device,
            descriptor: inputMetaDescriptor)

        inputMetaArray.writeBytes(inputMetaPointer)

        let maskShape = InputShape.create(
            batchSize: batchSize as NSNumber,
            numChannels: 1,
            nnYLen: nnYLen,
            nnXLen: nnXLen)

        let maskDescriptor = MPSNDArrayDescriptor(
            dataType: mask.tensor.dataType,
            shape: maskShape)

        let maskArray = MPSNDArray(
            device: device,
            descriptor: maskDescriptor)

        var maskStrideArray = [
            MemoryLayout<Float32>.size,
            nnXLen.intValue * MemoryLayout<Float32>.size,
            nnYLen.intValue * nnXLen.intValue * MemoryLayout<Float32>.size,
            numInputChannels.intValue * nnYLen.intValue * nnXLen.intValue
                * MemoryLayout<Float32>.size,
        ]

        maskArray.writeBytes(inputPointer, strideBytes: &maskStrideArray)

        let feeds = [
            input.tensor: MPSGraphTensorData(inputArray),
            inputGlobal.tensor: MPSGraphTensorData(inputGlobalArray),
            inputMeta.tensor: MPSGraphTensorData(inputMetaArray),
            mask.tensor: MPSGraphTensorData(maskArray),
        ]

        let fetch = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: targetTensors,
            targetOperations: nil)

        assert(fetch[policyHead.policyTensor] != nil)
        assert(fetch[policyHead.policyPassTensor] != nil)
        assert(fetch[valueHead.valueTensor] != nil)
        assert(fetch[valueHead.scoreValueTensor] != nil)
        assert(fetch[valueHead.ownershipTensor] != nil)

        fetch[policyHead.policyTensor]?.mpsndarray().readBytes(policy)
        fetch[policyHead.policyPassTensor]?.mpsndarray().readBytes(policyPass)
        fetch[valueHead.valueTensor]?.mpsndarray().readBytes(value)
        fetch[valueHead.scoreValueTensor]?.mpsndarray().readBytes(scoreValue)
        fetch[valueHead.ownershipTensor]?.mpsndarray().readBytes(ownership)
    }
}

public struct SWTransformerRMSNormDesc {
    let numChannels: NSNumber
    let weights: UnsafeMutablePointer<Float32>

    init(
        numChannels: NSNumber,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.numChannels = numChannels
        self.weights = weights
    }
}

public func createSWTransformerRMSNormDesc(
    numChannels: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWTransformerRMSNormDesc {
    return SWTransformerRMSNormDesc(
        numChannels: numChannels as NSNumber,
        weights: weights)
}

public struct SWTransformerBlockDesc {
    let norm1: SWTransformerRMSNormDesc
    let qProj: SWMatMulLayerDesc
    let kProj: SWMatMulLayerDesc
    let vProj: SWMatMulLayerDesc
    let outProj: SWMatMulLayerDesc
    let norm2: SWTransformerRMSNormDesc
    let ffnW1: SWMatMulLayerDesc
    let ffnWGate: SWMatMulLayerDesc
    let ffnW2: SWMatMulLayerDesc
}

public func createSWTransformerBlockDesc(
    norm1: SWTransformerRMSNormDesc,
    qProj: SWMatMulLayerDesc,
    kProj: SWMatMulLayerDesc,
    vProj: SWMatMulLayerDesc,
    outProj: SWMatMulLayerDesc,
    norm2: SWTransformerRMSNormDesc,
    ffnW1: SWMatMulLayerDesc,
    ffnWGate: SWMatMulLayerDesc,
    ffnW2: SWMatMulLayerDesc
) -> SWTransformerBlockDesc {
    return SWTransformerBlockDesc(
        norm1: norm1,
        qProj: qProj,
        kProj: kProj,
        vProj: vProj,
        outProj: outProj,
        norm2: norm2,
        ffnW1: ffnW1,
        ffnWGate: ffnWGate,
        ffnW2: ffnW2)
}

public class TransformerBlockDescBuilder {
    public var blockDescriptors: [SWTransformerBlockDesc] = []

    public func enque(with descriptor: SWTransformerBlockDesc) {
        blockDescriptors.append(descriptor)
    }
}

public func createTransformerBlockDescBuilder() -> TransformerBlockDescBuilder {
    return TransformerBlockDescBuilder()
}

public struct SWTransformerModelDesc {
    let version: Int
    let name: String
    let posLen: NSNumber
    let hiddenSize: NSNumber
    let numHeads: NSNumber
    let headDim: NSNumber
    let ffnDim: NSNumber
    let numInputChannels: NSNumber
    let numInputGlobalChannels: NSNumber
    let stemKernelSize: NSNumber
    let hasPosEmbed: Bool
    let scoreMode: Int
    let numScoreBeliefs: NSNumber
    let scoreBeliefLen: NSNumber
    let stemConv: SWConvLayerDesc
    let stemGlobal: SWMatMulLayerDesc
    let posEmbed: UnsafeMutablePointer<Float32>?
    let ropeCos: UnsafeMutablePointer<Float32>
    let ropeSin: UnsafeMutablePointer<Float32>
    let blocks: [SWTransformerBlockDesc]
    let finalNorm: SWTransformerRMSNormDesc
    let policyBoard: SWMatMulLayerDesc
    let policyPass: SWMatMulLayerDesc
    let policyBoardFull: SWMatMulLayerDesc
    let policyPassFull: SWMatMulLayerDesc
    let value: SWMatMulLayerDesc
    let misc: SWMatMulLayerDesc
    let moreMisc: SWMatMulLayerDesc
    let scoreValue: SWMatMulLayerDesc
    let ownership: SWMatMulLayerDesc
    let scoring: SWMatMulLayerDesc
    let futurePos: SWMatMulLayerDesc
    let seki: SWMatMulLayerDesc
    let scoreBelief: SWMatMulLayerDesc
    let scoreBeliefS2OffWeight: UnsafeMutablePointer<Float32>?
    let scoreBeliefS2ParWeight: UnsafeMutablePointer<Float32>?

    var scoreBeliefProjectSize: Int {
        if scoreMode == 0 {
            return scoreBeliefLen.intValue
        }
        return (scoreBeliefLen.intValue * numScoreBeliefs.intValue) + numScoreBeliefs.intValue
    }

    var supportsFullRawOutputs: Bool {
        return policyBoardFull.outChannels.intValue > 0 &&
            policyPassFull.outChannels.intValue > 0 &&
            misc.outChannels.intValue > 0 &&
            moreMisc.outChannels.intValue > 0 &&
            scoring.outChannels.intValue > 0 &&
            futurePos.outChannels.intValue > 0 &&
            seki.outChannels.intValue > 0 &&
            scoreBelief.outChannels.intValue > 0
    }
}

public func createSWTransformerModelDesc(
    version: Int32,
    name: String,
    posLen: Int32,
    hiddenSize: Int32,
    numHeads: Int32,
    headDim: Int32,
    ffnDim: Int32,
    numInputChannels: Int32,
    numInputGlobalChannels: Int32,
    stemKernelSize: Int32,
    hasPosEmbed: Bool,
    scoreMode: Int32,
    numScoreBeliefs: Int32,
    scoreBeliefLen: Int32,
    stemConv: SWConvLayerDesc,
    stemGlobal: SWMatMulLayerDesc,
    posEmbed: UnsafeMutablePointer<Float32>?,
    ropeCos: UnsafeMutablePointer<Float32>,
    ropeSin: UnsafeMutablePointer<Float32>,
    blocks: [SWTransformerBlockDesc],
    finalNorm: SWTransformerRMSNormDesc,
    policyBoard: SWMatMulLayerDesc,
    policyPass: SWMatMulLayerDesc,
    policyBoardFull: SWMatMulLayerDesc,
    policyPassFull: SWMatMulLayerDesc,
    value: SWMatMulLayerDesc,
    misc: SWMatMulLayerDesc,
    moreMisc: SWMatMulLayerDesc,
    scoreValue: SWMatMulLayerDesc,
    ownership: SWMatMulLayerDesc,
    scoring: SWMatMulLayerDesc,
    futurePos: SWMatMulLayerDesc,
    seki: SWMatMulLayerDesc,
    scoreBelief: SWMatMulLayerDesc,
    scoreBeliefS2OffWeight: UnsafeMutablePointer<Float32>?,
    scoreBeliefS2ParWeight: UnsafeMutablePointer<Float32>?
) -> SWTransformerModelDesc {
    return SWTransformerModelDesc(
        version: Int(version),
        name: name,
        posLen: posLen as NSNumber,
        hiddenSize: hiddenSize as NSNumber,
        numHeads: numHeads as NSNumber,
        headDim: headDim as NSNumber,
        ffnDim: ffnDim as NSNumber,
        numInputChannels: numInputChannels as NSNumber,
        numInputGlobalChannels: numInputGlobalChannels as NSNumber,
        stemKernelSize: stemKernelSize as NSNumber,
        hasPosEmbed: hasPosEmbed,
        scoreMode: Int(scoreMode),
        numScoreBeliefs: numScoreBeliefs as NSNumber,
        scoreBeliefLen: scoreBeliefLen as NSNumber,
        stemConv: stemConv,
        stemGlobal: stemGlobal,
        posEmbed: posEmbed,
        ropeCos: ropeCos,
        ropeSin: ropeSin,
        blocks: blocks,
        finalNorm: finalNorm,
        policyBoard: policyBoard,
        policyPass: policyPass,
        policyBoardFull: policyBoardFull,
        policyPassFull: policyPassFull,
        value: value,
        misc: misc,
        moreMisc: moreMisc,
        scoreValue: scoreValue,
        ownership: ownership,
        scoring: scoring,
        futurePos: futurePos,
        seki: seki,
        scoreBelief: scoreBelief,
        scoreBeliefS2OffWeight: scoreBeliefS2OffWeight,
        scoreBeliefS2ParWeight: scoreBeliefS2ParWeight)
}

enum TransformerComputePrecision: String {
    case float32 = "fp32"
    case float16 = "fp16"
    case bfloat16 = "bf16"

    var dataType: MPSDataType {
        switch self {
        case .float32:
            return .float32
        case .float16:
            return .float16
        case .bfloat16:
            if #available(macOS 14.0, *) {
                return .bFloat16
            }
            return .float32
        }
    }
}

private func deviceSupportsTransformerBFloat16(_ device: MTLDevice) -> Bool {
    guard #available(macOS 14.0, *) else {
        return false
    }
    return device.supportsFamily(.apple7) ||
        device.supportsFamily(.apple8) ||
        device.supportsFamily(.apple9) ||
        device.supportsFamily(.apple10) ||
        device.supportsFamily(.mac2)
}

public func metalDeviceSupportsTransformerBFloat16() -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return false
    }
    return deviceSupportsTransformerBFloat16(device)
}

private func chooseTransformerPrecision(
    device: MTLDevice,
    requestedModeRaw: Int32,
    fallbackFP16Mode: SWEnable
) -> TransformerComputePrecision {
    switch requestedModeRaw {
    case 0:
        return .float32
    case 1:
        return .float16
    case 2:
        precondition(deviceSupportsTransformerBFloat16(device), "Metal backend: 当前设备不支持 Transformer bf16 推理")
        return .bfloat16
    case 3:
        switch fallbackFP16Mode {
        case .False:
            return .float32
        case .True:
            return .float16
        case .Auto:
            return deviceSupportsTransformerBFloat16(device) ? .bfloat16 : .float16
        }
    default:
        return deviceSupportsTransformerBFloat16(device) ? .bfloat16 : .float16
    }
}

private func makeConstantTensor(
    graph: MPSGraph,
    pointer: UnsafeMutablePointer<Float32>,
    shape: [NSNumber],
    dataType: MPSDataType = .float32
) -> MPSGraphTensor {
    let data = Data(floatsNoCopy: pointer, shape: shape)
    let tensor = graph.constant(data, shape: shape, dataType: .float32)
    return castTensorIfNeeded(graph: graph, sourceTensor: tensor, to: dataType)
}

private func swapAxes(
    graph: MPSGraph,
    sourceTensor: MPSGraphTensor,
    axis1: Int,
    axis2: Int
) -> MPSGraphTensor {
    return graph.transposeTensor(sourceTensor, dimension: axis1, withDimension: axis2, name: nil)
}

private func transposeNHWCToNCHW(
    graph: MPSGraph,
    sourceTensor: MPSGraphTensor
) -> MPSGraphTensor {
    let ncwh = swapAxes(graph: graph, sourceTensor: sourceTensor, axis1: 1, axis2: 3)
    return swapAxes(graph: graph, sourceTensor: ncwh, axis1: 2, axis2: 3)
}

private func transposeNCHWToNHWC(
    graph: MPSGraph,
    sourceTensor: MPSGraphTensor
) -> MPSGraphTensor {
    let nhcw = swapAxes(graph: graph, sourceTensor: sourceTensor, axis1: 1, axis2: 2)
    return swapAxes(graph: graph, sourceTensor: nhcw, axis1: 2, axis2: 3)
}

private func transformerLinear(
    graph: MPSGraph,
    descriptor: SWMatMulLayerDesc,
    sourceTensor: MPSGraphTensor,
    outputShape: [NSNumber],
    computationDataType: MPSDataType? = nil,
    outputDataType: MPSDataType? = nil
) -> MPSGraphTensor {
    precondition(descriptor.weights != nil)
    let opDataType = computationDataType ?? sourceTensor.dataType
    let weightsTensor = makeConstantTensor(
        graph: graph,
        pointer: descriptor.weights!,
        shape: [descriptor.inChannels, descriptor.outChannels],
        dataType: opDataType)
    let flatInput = graph.reshape(
        castTensorIfNeeded(graph: graph, sourceTensor: sourceTensor, to: opDataType),
        shape: [-1, descriptor.inChannels],
        name: nil)
    let flatOutput = graph.matrixMultiplication(primary: flatInput, secondary: weightsTensor, name: nil)
    let reshaped = graph.reshape(flatOutput, shape: outputShape, name: nil)
    return castTensorIfNeeded(
        graph: graph,
        sourceTensor: reshaped,
        to: outputDataType ?? sourceTensor.dataType)
}

private func transformerRMSNorm(
    graph: MPSGraph,
    sourceTensor: MPSGraphTensor,
    descriptor: SWTransformerRMSNormDesc
) -> MPSGraphTensor {
    let sourceFP32 = castTensorIfNeeded(graph: graph, sourceTensor: sourceTensor, to: .float32)
    let squared = graph.square(with: sourceFP32, name: nil)
    let sum = graph.reductionSum(with: squared, axes: [2], name: nil)
    let denom = graph.constant(Double(descriptor.numChannels.intValue), dataType: .float32)
    let mean = graph.division(sum, denom, name: nil)
    let eps = graph.constant(1e-6, dataType: .float32)
    let variance = graph.addition(mean, eps, name: nil)
    let rms = graph.squareRoot(with: variance, name: nil)
    let one = graph.constant(1.0, dataType: .float32)
    let invRms = graph.division(one, rms, name: nil)
    let normalized = graph.multiplication(sourceFP32, invRms, name: nil)
    let weightTensor = makeConstantTensor(
        graph: graph,
        pointer: descriptor.weights,
        shape: [1, 1, descriptor.numChannels],
        dataType: .float32)
    let scaled = graph.multiplication(normalized, weightTensor, name: nil)
    return castTensorIfNeeded(graph: graph, sourceTensor: scaled, to: sourceTensor.dataType)
}

private func transformerMeanPool(
    graph: MPSGraph,
    sourceTensor: MPSGraphTensor,
    seqLen: Int
) -> MPSGraphTensor {
    let sum = graph.reductionSum(with: sourceTensor, axes: [1], name: nil)
    let denom = graph.constant(Double(seqLen), dataType: sourceTensor.dataType)
    let mean = graph.division(sum, denom, name: nil)
    return graph.reshape(mean, shape: [-1, sourceTensor.shape![2]], name: nil)
}

private func transformerApplyRoPE(
    graph: MPSGraph,
    sourceTensor: MPSGraphTensor,
    ropeCosTensor: MPSGraphTensor,
    ropeSinTensor: MPSGraphTensor
) -> MPSGraphTensor {
    let sourceFP32 = castTensorIfNeeded(graph: graph, sourceTensor: sourceTensor, to: .float32)
    let halves = graph.split(sourceFP32, numSplits: 2, axis: 3, name: nil)
    let minusOne = graph.constant(-1.0, dataType: .float32)
    let negUpper = graph.multiplication(halves[1], minusOne, name: nil)
    let rotated = graph.concatTensors([negUpper, halves[0]], dimension: 3, name: nil)
    let mulCos = graph.multiplication(sourceFP32, castTensorIfNeeded(graph: graph, sourceTensor: ropeCosTensor, to: .float32), name: nil)
    let mulSin = graph.multiplication(rotated, castTensorIfNeeded(graph: graph, sourceTensor: ropeSinTensor, to: .float32), name: nil)
    let rotatedOut = graph.addition(mulCos, mulSin, name: nil)
    return castTensorIfNeeded(graph: graph, sourceTensor: rotatedOut, to: sourceTensor.dataType)
}

private func transformerSwiGLU(
    graph: MPSGraph,
    lhs: MPSGraphTensor,
    rhs: MPSGraphTensor
) -> MPSGraphTensor {
    let sigmoid = graph.sigmoid(with: lhs, name: nil)
    let silu = graph.multiplication(lhs, sigmoid, name: nil)
    let siluFP32 = castTensorIfNeeded(graph: graph, sourceTensor: silu, to: .float32)
    let rhsFP32 = castTensorIfNeeded(graph: graph, sourceTensor: rhs, to: .float32)
    let multiplied = graph.multiplication(siluFP32, rhsFP32, name: nil)
    return castTensorIfNeeded(graph: graph, sourceTensor: multiplied, to: lhs.dataType)
}

private func transformerAdd(
    graph: MPSGraph,
    lhs: MPSGraphTensor,
    rhs: MPSGraphTensor,
    resultDataType: MPSDataType
) -> MPSGraphTensor {
    return graph.addition(
        castTensorIfNeeded(graph: graph, sourceTensor: lhs, to: resultDataType),
        castTensorIfNeeded(graph: graph, sourceTensor: rhs, to: resultDataType),
        name: nil)
}

private func transformerScoreBeliefLogSoftmax(_ values: inout [Float32]) {
    guard !values.isEmpty else {
        return
    }
    var maxValue = values[0]
    for value in values.dropFirst() {
        maxValue = Swift.max(maxValue, value)
    }
    var sum: Double = 0.0
    for value in values {
        sum += Foundation.exp(Double(value - maxValue))
    }
    let logSum = Float32(Double(maxValue) + Foundation.log(sum))
    for index in values.indices {
        values[index] -= logSum
    }
}

struct TransformerModel {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let graph: MPSGraph
    let desc: SWTransformerModelDesc
    let precision: TransformerComputePrecision
    let computeDataType: MPSDataType
    let inputTensor: MPSGraphTensor
    let inputGlobalTensor: MPSGraphTensor
    let policyTensor: MPSGraphTensor
    let policyPassTensor: MPSGraphTensor
    let valueTensor: MPSGraphTensor
    let scoreValueTensor: MPSGraphTensor
    let ownershipTensor: MPSGraphTensor
    let fullPolicyTensor: MPSGraphTensor?
    let fullPolicyPassTensor: MPSGraphTensor?
    let miscTensor: MPSGraphTensor?
    let moreMiscTensor: MPSGraphTensor?
    let scoringTensor: MPSGraphTensor?
    let futurePosTensor: MPSGraphTensor?
    let sekiTensor: MPSGraphTensor?
    let scoreBeliefProjectTensor: MPSGraphTensor?
    let subsetTargets: [MPSGraphTensor]
    let fullTargets: [MPSGraphTensor]

    init(
        device: MTLDevice,
        graph: MPSGraph,
        descriptor: SWTransformerModelDesc,
        precision: TransformerComputePrecision
    ) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.graph = graph
        self.desc = descriptor
        self.precision = precision
        self.computeDataType = precision.dataType

        inputTensor = graph.placeholder(
            shape: [-1, descriptor.posLen, descriptor.posLen, descriptor.numInputChannels],
            dataType: .float32,
            name: nil)
        inputGlobalTensor = graph.placeholder(
            shape: [-1, descriptor.numInputGlobalChannels],
            dataType: .float32,
            name: nil)

        let inputCompute = castTensorIfNeeded(graph: graph, sourceTensor: inputTensor, to: computeDataType)
        let inputGlobalCompute = castTensorIfNeeded(graph: graph, sourceTensor: inputGlobalTensor, to: computeDataType)
        let inputNCHW = transposeNHWCToNCHW(graph: graph, sourceTensor: inputCompute)
        let stableLinearDataType = computeDataType
        let stemConv = ConvLayer(
            graph: graph,
            sourceTensor: inputNCHW,
            descriptor: descriptor.stemConv,
            nnXLen: descriptor.posLen,
            nnYLen: descriptor.posLen,
            computationDataType: stableLinearDataType,
            outputDataType: computeDataType)
        let stemSpatialNHWC = transposeNCHWToNHWC(graph: graph, sourceTensor: stemConv.resultTensor)
        let seqLen = descriptor.posLen.intValue * descriptor.posLen.intValue
        let numHeads = descriptor.numHeads.intValue
        let headDim = descriptor.headDim.intValue
        let ropeShape: [NSNumber] = [1, seqLen as NSNumber, 1, descriptor.headDim]

        var x = graph.reshape(stemSpatialNHWC, shape: [-1, seqLen as NSNumber, descriptor.hiddenSize], name: nil)
        let stemGlobal = transformerLinear(
            graph: graph,
            descriptor: descriptor.stemGlobal,
            sourceTensor: inputGlobalCompute,
            outputShape: [-1, descriptor.hiddenSize],
            computationDataType: stableLinearDataType)
        let stemGlobalBroadcast = graph.reshape(stemGlobal, shape: [-1, 1, descriptor.hiddenSize], name: nil)
        x = transformerAdd(graph: graph, lhs: x, rhs: stemGlobalBroadcast, resultDataType: computeDataType)

        if descriptor.hasPosEmbed {
            let posTensor = makeConstantTensor(
                graph: graph,
                pointer: descriptor.posEmbed!,
                shape: [1, seqLen as NSNumber, descriptor.hiddenSize],
                dataType: computeDataType)
            x = transformerAdd(graph: graph, lhs: x, rhs: posTensor, resultDataType: computeDataType)
        }

        let ropeCosTensor = makeConstantTensor(
            graph: graph,
            pointer: descriptor.ropeCos,
            shape: ropeShape,
            dataType: .float32)
        let ropeSinTensor = makeConstantTensor(
            graph: graph,
            pointer: descriptor.ropeSin,
            shape: ropeShape,
            dataType: .float32)

        for block in descriptor.blocks {
            let norm1 = transformerRMSNorm(graph: graph, sourceTensor: x, descriptor: block.norm1)
            var q = transformerLinear(
                graph: graph,
                descriptor: block.qProj,
                sourceTensor: norm1,
                outputShape: [-1, seqLen as NSNumber, descriptor.hiddenSize],
                computationDataType: stableLinearDataType)
            var k = transformerLinear(
                graph: graph,
                descriptor: block.kProj,
                sourceTensor: norm1,
                outputShape: [-1, seqLen as NSNumber, descriptor.hiddenSize],
                computationDataType: stableLinearDataType)
            let v = transformerLinear(
                graph: graph,
                descriptor: block.vProj,
                sourceTensor: norm1,
                outputShape: [-1, seqLen as NSNumber, descriptor.hiddenSize],
                computationDataType: stableLinearDataType)

            q = graph.reshape(q, shape: [-1, seqLen as NSNumber, numHeads as NSNumber, headDim as NSNumber], name: nil)
            k = graph.reshape(k, shape: [-1, seqLen as NSNumber, numHeads as NSNumber, headDim as NSNumber], name: nil)
            let vHeads = graph.reshape(v, shape: [-1, seqLen as NSNumber, numHeads as NSNumber, headDim as NSNumber], name: nil)

            q = transformerApplyRoPE(graph: graph, sourceTensor: q, ropeCosTensor: ropeCosTensor, ropeSinTensor: ropeSinTensor)
            k = transformerApplyRoPE(graph: graph, sourceTensor: k, ropeCosTensor: ropeCosTensor, ropeSinTensor: ropeSinTensor)

            let qAttention = swapAxes(graph: graph, sourceTensor: q, axis1: 1, axis2: 2)
            let kAttention = swapAxes(graph: graph, sourceTensor: k, axis1: 1, axis2: 2)
            let vAttention = swapAxes(graph: graph, sourceTensor: vHeads, axis1: 1, axis2: 2)
            let attn = graph.scaledDotProductAttention(
                query: qAttention,
                key: kAttention,
                value: vAttention,
                scale: Float(1.0 / Foundation.sqrt(Double(headDim))),
                name: nil)
            let attnTransposed = swapAxes(graph: graph, sourceTensor: attn, axis1: 1, axis2: 2)
            let attnFlat = graph.reshape(attnTransposed, shape: [-1, seqLen as NSNumber, descriptor.hiddenSize], name: nil)
            let attnProj = transformerLinear(
                graph: graph,
                descriptor: block.outProj,
                sourceTensor: attnFlat,
                outputShape: [-1, seqLen as NSNumber, descriptor.hiddenSize],
                computationDataType: stableLinearDataType)
            x = transformerAdd(graph: graph, lhs: x, rhs: attnProj, resultDataType: computeDataType)

            let norm2 = transformerRMSNorm(graph: graph, sourceTensor: x, descriptor: block.norm2)
            let ffn1 = transformerLinear(
                graph: graph,
                descriptor: block.ffnW1,
                sourceTensor: norm2,
                outputShape: [-1, seqLen as NSNumber, descriptor.ffnDim],
                computationDataType: stableLinearDataType)
            let ffnGate = transformerLinear(
                graph: graph,
                descriptor: block.ffnWGate,
                sourceTensor: norm2,
                outputShape: [-1, seqLen as NSNumber, descriptor.ffnDim],
                computationDataType: stableLinearDataType)
            let ffnAct = transformerSwiGLU(graph: graph, lhs: ffn1, rhs: ffnGate)
            let ffnOut = transformerLinear(
                graph: graph,
                descriptor: block.ffnW2,
                sourceTensor: ffnAct,
                outputShape: [-1, seqLen as NSNumber, descriptor.hiddenSize],
                computationDataType: stableLinearDataType)
            x = transformerAdd(graph: graph, lhs: x, rhs: ffnOut, resultDataType: computeDataType)
        }

        let xFinal = transformerRMSNorm(graph: graph, sourceTensor: x, descriptor: descriptor.finalNorm)
        let pooled = transformerMeanPool(graph: graph, sourceTensor: xFinal, seqLen: seqLen)

        policyTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
            graph: graph,
            descriptor: descriptor.policyBoard,
            sourceTensor: xFinal,
            outputShape: [-1, seqLen as NSNumber, descriptor.policyBoard.outChannels],
            computationDataType: stableLinearDataType), to: .float32)
        policyPassTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
            graph: graph,
            descriptor: descriptor.policyPass,
            sourceTensor: pooled,
            outputShape: [-1, descriptor.policyPass.outChannels],
            computationDataType: stableLinearDataType), to: .float32)
        valueTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
            graph: graph,
            descriptor: descriptor.value,
            sourceTensor: pooled,
            outputShape: [-1, descriptor.value.outChannels],
            computationDataType: stableLinearDataType), to: .float32)
        scoreValueTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
            graph: graph,
            descriptor: descriptor.scoreValue,
            sourceTensor: pooled,
            outputShape: [-1, descriptor.scoreValue.outChannels],
            computationDataType: stableLinearDataType), to: .float32)
        let ownershipExpanded = transformerLinear(
            graph: graph,
            descriptor: descriptor.ownership,
            sourceTensor: xFinal,
            outputShape: [-1, seqLen as NSNumber, 1],
            computationDataType: stableLinearDataType)
        ownershipTensor = castTensorIfNeeded(
            graph: graph,
            sourceTensor: graph.reshape(ownershipExpanded, shape: [-1, seqLen as NSNumber], name: nil),
            to: .float32)

        if descriptor.supportsFullRawOutputs {
            fullPolicyTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
                graph: graph,
                descriptor: descriptor.policyBoardFull,
                sourceTensor: xFinal,
                outputShape: [-1, seqLen as NSNumber, descriptor.policyBoardFull.outChannels],
                computationDataType: stableLinearDataType), to: .float32)
            fullPolicyPassTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
                graph: graph,
                descriptor: descriptor.policyPassFull,
                sourceTensor: pooled,
                outputShape: [-1, descriptor.policyPassFull.outChannels],
                computationDataType: stableLinearDataType), to: .float32)
            miscTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
                graph: graph,
                descriptor: descriptor.misc,
                sourceTensor: pooled,
                outputShape: [-1, descriptor.misc.outChannels],
                computationDataType: stableLinearDataType), to: .float32)
            moreMiscTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
                graph: graph,
                descriptor: descriptor.moreMisc,
                sourceTensor: pooled,
                outputShape: [-1, descriptor.moreMisc.outChannels],
                computationDataType: stableLinearDataType), to: .float32)
            let scoringExpanded = transformerLinear(
                graph: graph,
                descriptor: descriptor.scoring,
                sourceTensor: xFinal,
                outputShape: [-1, seqLen as NSNumber, 1],
                computationDataType: stableLinearDataType)
            scoringTensor = castTensorIfNeeded(
                graph: graph,
                sourceTensor: graph.reshape(scoringExpanded, shape: [-1, seqLen as NSNumber], name: nil),
                to: .float32)
            futurePosTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
                graph: graph,
                descriptor: descriptor.futurePos,
                sourceTensor: xFinal,
                outputShape: [-1, seqLen as NSNumber, descriptor.futurePos.outChannels],
                computationDataType: stableLinearDataType), to: .float32)
            sekiTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
                graph: graph,
                descriptor: descriptor.seki,
                sourceTensor: xFinal,
                outputShape: [-1, seqLen as NSNumber, descriptor.seki.outChannels],
                computationDataType: stableLinearDataType), to: .float32)
            scoreBeliefProjectTensor = castTensorIfNeeded(graph: graph, sourceTensor: transformerLinear(
                graph: graph,
                descriptor: descriptor.scoreBelief,
                sourceTensor: pooled,
                outputShape: [-1, descriptor.scoreBelief.outChannels],
                computationDataType: stableLinearDataType), to: .float32)
            fullTargets = [
                fullPolicyPassTensor!,
                fullPolicyTensor!,
                valueTensor,
                miscTensor!,
                moreMiscTensor!,
                ownershipTensor,
                scoringTensor!,
                futurePosTensor!,
                sekiTensor!,
                scoreBeliefProjectTensor!,
            ]
        } else {
            fullPolicyTensor = nil
            fullPolicyPassTensor = nil
            miscTensor = nil
            moreMiscTensor = nil
            scoringTensor = nil
            futurePosTensor = nil
            sekiTensor = nil
            scoreBeliefProjectTensor = nil
            fullTargets = []
        }

        subsetTargets = [
            policyPassTensor,
            policyTensor,
            valueTensor,
            scoreValueTensor,
            ownershipTensor,
        ]
    }

    private func makeInputArray(
        pointer: UnsafeMutablePointer<Float32>,
        shape: [NSNumber]
    ) -> MPSNDArray {
        let descriptor = MPSNDArrayDescriptor(dataType: .float32, shape: shape)
        let array = MPSNDArray(device: device, descriptor: descriptor)
        array.writeBytes(pointer)
        return array
    }

    private func run(
        batchSize: Int,
        targetTensors: [MPSGraphTensor],
        inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobalPointer: UnsafeMutablePointer<Float32>
    ) -> [MPSGraphTensor : MPSGraphTensorData] {
        let inputArray = makeInputArray(
            pointer: inputPointer,
            shape: [batchSize as NSNumber, desc.posLen, desc.posLen, desc.numInputChannels])
        let inputGlobalArray = makeInputArray(
            pointer: inputGlobalPointer,
            shape: [batchSize as NSNumber, desc.numInputGlobalChannels])
        let feeds = [
            inputTensor: MPSGraphTensorData(inputArray),
            inputGlobalTensor: MPSGraphTensorData(inputGlobalArray),
        ]
        return graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: targetTensors,
            targetOperations: nil)
    }

    private func finalizeScoreBelief(
        inputGlobal: UnsafeMutablePointer<Float32>,
        project: UnsafeMutablePointer<Float32>,
        output: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {
        let scoreBeliefLen = desc.scoreBeliefLen.intValue
        let projectSize = desc.scoreBeliefProjectSize
        let numBeliefs = desc.numScoreBeliefs.intValue
        let numGlobal = desc.numInputGlobalChannels.intValue

        for batchIdx in 0 ..< batchSize {
            let projectBase = batchIdx * projectSize
            let outputBase = batchIdx * scoreBeliefLen
            if desc.scoreMode == 0 {
                var logits = Array(UnsafeBufferPointer(start: project + projectBase, count: scoreBeliefLen))
                transformerScoreBeliefLogSoftmax(&logits)
                for index in 0 ..< scoreBeliefLen {
                    output[outputBase + index] = logits[index]
                }
                continue
            }

            var belief = Array(
                UnsafeBufferPointer(
                    start: project + projectBase,
                    count: scoreBeliefLen * numBeliefs))
            var mixLogits = Array(
                UnsafeBufferPointer(
                    start: project + projectBase + scoreBeliefLen * numBeliefs,
                    count: numBeliefs))

            if desc.scoreMode == 2 {
                let scoreParity = inputGlobal[batchIdx * numGlobal + numGlobal - 1]
                let mid = scoreBeliefLen / 2
                for i in 0 ..< scoreBeliefLen {
                    let diff = i - mid
                    let parityBit = ((diff % 2) + 2) % 2
                    let offsetTerm = Float32(0.05) * (Float32(diff) + Float32(0.5))
                    let parityTerm = (Float32(0.5) - Float32(parityBit)) * scoreParity
                    for j in 0 ..< numBeliefs {
                        belief[i * numBeliefs + j] +=
                            offsetTerm * desc.scoreBeliefS2OffWeight![j] +
                            parityTerm * desc.scoreBeliefS2ParWeight![j]
                    }
                }
            }

            for beliefIdx in 0 ..< numBeliefs {
                var slice = [Float32](repeating: 0.0, count: scoreBeliefLen)
                for scoreIdx in 0 ..< scoreBeliefLen {
                    slice[scoreIdx] = belief[scoreIdx * numBeliefs + beliefIdx]
                }
                transformerScoreBeliefLogSoftmax(&slice)
                for scoreIdx in 0 ..< scoreBeliefLen {
                    belief[scoreIdx * numBeliefs + beliefIdx] = slice[scoreIdx]
                }
            }

            transformerScoreBeliefLogSoftmax(&mixLogits)
            for scoreIdx in 0 ..< scoreBeliefLen {
                var maxValue = belief[scoreIdx * numBeliefs] + mixLogits[0]
                for beliefIdx in 1 ..< numBeliefs {
                    maxValue = Swift.max(maxValue, belief[scoreIdx * numBeliefs + beliefIdx] + mixLogits[beliefIdx])
                }
                var sum: Double = 0.0
                for beliefIdx in 0 ..< numBeliefs {
                    sum += Foundation.exp(
                        Double(belief[scoreIdx * numBeliefs + beliefIdx] + mixLogits[beliefIdx] - maxValue))
                }
                output[outputBase + scoreIdx] = Float32(Double(maxValue) + Foundation.log(sum))
            }
        }
    }

    func apply(
        input inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {
        let fetch = run(
            batchSize: batchSize,
            targetTensors: subsetTargets,
            inputPointer: inputPointer,
            inputGlobalPointer: inputGlobalPointer)
        fetch[policyPassTensor]?.mpsndarray().readBytes(policyPass)
        fetch[policyTensor]?.mpsndarray().readBytes(policy)
        fetch[valueTensor]?.mpsndarray().readBytes(value)
        fetch[scoreValueTensor]?.mpsndarray().readBytes(scoreValue)
        fetch[ownershipTensor]?.mpsndarray().readBytes(ownership)
    }

    func applyFull(
        input inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        misc: UnsafeMutablePointer<Float32>,
        moreMisc: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        scoring: UnsafeMutablePointer<Float32>,
        futurePos: UnsafeMutablePointer<Float32>,
        seki: UnsafeMutablePointer<Float32>,
        scoreBelief: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {
        precondition(desc.supportsFullRawOutputs)
        let fetch = run(
            batchSize: batchSize,
            targetTensors: fullTargets,
            inputPointer: inputPointer,
            inputGlobalPointer: inputGlobalPointer)
        fetch[fullPolicyPassTensor!]?.mpsndarray().readBytes(policyPass)
        fetch[fullPolicyTensor!]?.mpsndarray().readBytes(policy)
        fetch[valueTensor]?.mpsndarray().readBytes(value)
        fetch[miscTensor!]?.mpsndarray().readBytes(misc)
        fetch[moreMiscTensor!]?.mpsndarray().readBytes(moreMisc)
        fetch[ownershipTensor]?.mpsndarray().readBytes(ownership)
        fetch[scoringTensor!]?.mpsndarray().readBytes(scoring)
        fetch[futurePosTensor!]?.mpsndarray().readBytes(futurePos)
        fetch[sekiTensor!]?.mpsndarray().readBytes(seki)

        let projectCount = batchSize * desc.scoreBeliefProjectSize
        let projectBuffer = UnsafeMutablePointer<Float32>.allocate(capacity: projectCount)
        defer { projectBuffer.deallocate() }
        fetch[scoreBeliefProjectTensor!]?.mpsndarray().readBytes(projectBuffer)
        finalizeScoreBelief(
            inputGlobal: inputGlobalPointer,
            project: projectBuffer,
            output: scoreBelief,
            batchSize: batchSize)
    }
}

public class MetalTransformerComputeHandle {
    let model: TransformerModel

    init(model: TransformerModel) {
        self.model = model
    }

    public func apply(
        _ batchSize: Int,
        input inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>
    ) {
        autoreleasepool {
            model.apply(
                input: inputPointer,
                inputGlobal: inputGlobalPointer,
                policyPass: policyPass,
                policy: policy,
                value: value,
                scoreValue: scoreValue,
                ownership: ownership,
                batchSize: batchSize)
        }
    }

    public func applyFull(
        _ batchSize: Int,
        input inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        misc: UnsafeMutablePointer<Float32>,
        moreMisc: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        scoring: UnsafeMutablePointer<Float32>,
        futurePos: UnsafeMutablePointer<Float32>,
        seki: UnsafeMutablePointer<Float32>,
        scoreBelief: UnsafeMutablePointer<Float32>
    ) {
        autoreleasepool {
            model.applyFull(
                input: inputPointer,
                inputGlobal: inputGlobalPointer,
                policyPass: policyPass,
                policy: policy,
                value: value,
                misc: misc,
                moreMisc: moreMisc,
                ownership: ownership,
                scoring: scoring,
                futurePos: futurePos,
                seki: seki,
                scoreBelief: scoreBelief,
                batchSize: batchSize)
        }
    }
}

public func maybeCreateMetalTransformerComputeHandle(
    _ condition: Bool,
    _ serverThreadIdx: Int = 0,
    _ descriptor: SWTransformerModelDesc,
    _ context: MetalComputeContext
) -> MetalTransformerComputeHandle? {
    guard condition else { return nil }

    let device = MTLCreateSystemDefaultDevice()!
    let precision = chooseTransformerPrecision(
        device: device,
        requestedModeRaw: context.transformerPrecisionModeRaw,
        fallbackFP16Mode: context.useFP16Mode)
    let model = TransformerModel(
        device: device,
        graph: MPSGraph(),
        descriptor: descriptor,
        precision: precision)
    let handle = MetalTransformerComputeHandle(model: model)

    printError(
        "Metal backend \(serverThreadIdx): \(device.name), Transformer model version \(descriptor.version) \(descriptor.name), \(context.nnXLen)x\(context.nnYLen), precision \(precision.rawValue)"
    )
    if precision == .float16 {
        printError(
            "WARNING: Metal Transformer fp16 inference may overflow to NaN/Inf on some models or positions. This is more likely when the checkpoint was trained with bf16/fp32 or mixed precision. Validate raw outputs before relying on fp16."
        )
    }

    return handle
}

// A enum to represent enabled/disabled/auto option of a feature.
public enum SWEnable {
    case False
    case True
    case Auto
}

/// A class that represents context of GPU devices.
public class MetalComputeContext {
    public let nnXLen: Int32
    public let nnYLen: Int32
    let useFP16Mode: SWEnable
    let transformerPrecisionModeRaw: Int32
    let useNHWCMode: SWEnable

    /// Initialize a context.
    /// - Parameters:
    ///   - nnXLen: The width of the input tensor.
    ///   - nnYLen: The height of the input tensor.
    init(
        nnXLen: Int32,
        nnYLen: Int32,
        useFP16Mode: SWEnable,
        transformerPrecisionModeRaw: Int32,
        useNHWCMode: SWEnable
    ) {
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.useFP16Mode = useFP16Mode
        self.transformerPrecisionModeRaw = transformerPrecisionModeRaw
        self.useNHWCMode = useNHWCMode
    }
}

public func createMetalComputeContext(
    nnXLen: Int32,
    nnYLen: Int32,
    useFP16Mode: SWEnable,
    transformerPrecisionModeRaw: Int32,
    useNHWCMode: SWEnable
) -> MetalComputeContext {
    return MetalComputeContext(
        nnXLen: nnXLen,
        nnYLen: nnYLen,
        useFP16Mode: useFP16Mode,
        transformerPrecisionModeRaw: transformerPrecisionModeRaw,
        useNHWCMode: useNHWCMode)
}

/// A class that represents a handle of GPU device.
public class MetalComputeHandle {
    let model: Model

    init(model: Model) {
        self.model = model
    }

    public func apply(
        input inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
        inputMeta inputMetaPointer: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {
        autoreleasepool {
            model.apply(
                input: inputPointer,
                inputGlobal: inputGlobalPointer,
                inputMeta: inputMetaPointer,
                policy: policy,
                policyPass: policyPass,
                value: value,
                scoreValue: scoreValue,
                ownership: ownership,
                batchSize: batchSize)
        }
    }
}

public func maybeCreateMetalComputeHandle(
    condition: Bool,
    serverThreadIdx: Int = 0,
    descriptor: SWModelDesc,
    context: MetalComputeContext
) -> MetalComputeHandle? {
    guard condition else { return nil }

    let device = MTLCreateSystemDefaultDevice()!

    let model = Model(
        device: device,
        graph: MPSGraph(),
        descriptor: descriptor,
        nnXLen: context.nnXLen as NSNumber,
        nnYLen: context.nnYLen as NSNumber)

    let handle = MetalComputeHandle(model: model)

    printError(
        "Metal backend \(serverThreadIdx): \(device.name), Model version \(descriptor.version) \(descriptor.name), \(context.nnXLen)x\(context.nnYLen)"
    )

    return handle
}

public func printMetalDevices() {
    let device = MTLCreateSystemDefaultDevice()!
    printError("Found Metal Device: \(device.name)")
}
