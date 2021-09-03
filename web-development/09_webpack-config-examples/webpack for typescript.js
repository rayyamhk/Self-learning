const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const OptimizeCssAssetsPlugin = require("optimize-css-assets-webpack-plugin");
const TerserPlugin = require("terser-webpack-plugin");

module.exports = {
    mode: "production",
    entry: "./src/app.ts",
    output: {
        filename: "[contenthash].bundle.js",
        path: __dirname + "/build"
    },
    resolve: {
        //Rules for webpack resolution
        //if the path contains file extension, then ok
        //otherwise, webpack will resolve the path using resolve.extensions options
        //if different files have same name, then webpack will resolve them 
        //from right to left
        //Remark: it will override the default extensions,
        //i.e. for [".ts"], your js file will no longer be resolved
        extensions: [".js", ".ts"]
    },
    module: {
        rules: [
            {
                //You need to create extra tsconfig.json as configuration
                test: /\.ts$/,
                use: "ts-loader",
                exclude: /node_modules/
            },
            {
                test: /\.css$/,
                use: [MiniCssExtractPlugin.loader, "css-loader"]
            },
            {
                test: /\.(png|jpg|jpeg|svg|gif)/,
                use: {
                    loader: "file-loader",
                    options: {
                        name: "[hash].[name].[ext]",
                        outputPath: "img",
                        esModule: false
                    }
                }
            },
            {
                test: /\.html$/,
                use: "html-loader"
            }
        ]
    },
    plugins: [
        new MiniCssExtractPlugin({
            filename: "[hash].css"
        })
    ],
    optimization: {
        minimizer: [
            new HtmlWebpackPlugin({
                filename:"[hash].html",
                template: "./src/index.html",
                minify: {
                    removeAttributeQuotes: true,
                    collapseWhitespace: true,
                    removeComments: true
                }
            }),
            new TerserPlugin(),
            new OptimizeCssAssetsPlugin()
        ]
    }
}