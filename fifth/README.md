# fifth

Input uses csv_reader

Web packages added:
.package(name:"TensorBoardS",url: "https://github.com/t-ae/tensorboardS", .branch("master")),

>swift package update

Removed matplotlib and added "TensorBoardS" package

Using tensorboard to display per epoch:
1. Training loss
2. Training accuracy
3. Gradients 

We also define an extension to Dense() to display the count of any weights that went to zero.
