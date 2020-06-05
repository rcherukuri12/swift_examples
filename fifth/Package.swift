// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "fifth",
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(name:"readers",url:"https://github.com/rcherukuri12/data_readers", .branch("master")),
        .package(name:"TensorBoardS",url: "https://github.com/t-ae/tensorboardS", .branch("master")),
        .package(name:"TensorFlowModels", url: "https://github.com/tensorflow/swift-models/", .branch("tensorflow-0.9")),
        
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "fifth",
            dependencies: ["readers","TensorBoardS",]),
        .testTarget(
            name: "fifthTests",
            dependencies: ["fifth"]),
    ]
)
