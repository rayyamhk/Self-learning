const HtmlWebpackPlugin = require("html-webpack-plugin");

//package.json modifications:
//add a script "webpack-dev-server --config webpack.dev.js --open"
//it generates all files in memory and allows you to auto reload
module.exports = {
    mode: "development",
    entry: "./src/js/app.js", //support multiple entries
    output: {
        path: __dirname + "/dist", //need to be absolute, you can use path.resolve()
        filename: "app.js" //output js file name
    },
    module: { //define loaders
        rules: [
            {
                test: /\.css$/,
                //the priority is from right to left
                //sass-loader compile sass to css, which is optional
                //css-loader adds css content into entry js file
                //style-loader converts css content in js into style tag in html
                use: ["style-loader", "css-loader"]
            },
            {
                test: /\.html$/,
                //if you come across html file, it will convert all loadable attribute, e.g. src in <img>,
                //into import in the js file
                //example: <img src="./img/bg.jpg/> => import img from "./img/bg.jpg"
                //since webpack doesn't understand how to import img, we need file-loader
                use: "html-loader"
            },
            {
                test: /\.(jpg|jpeg|png|svg|gif)$/,
                use: {
                    loader: "file-loader",
                    options: {
                        name: "[name].[ext]",
                        outputPath: "img",
                        //it should be added
                        esModule: false
                    }
                }
            }
        ]
    },
    plugins: [
        //it takes an given html file as a template and add all dependencies needed.
        //if no template is defined, it will automatically generate a simple html
        new HtmlWebpackPlugin({
            filename: "index.html",
            template: "./src/index.html"
        })
    ]
}