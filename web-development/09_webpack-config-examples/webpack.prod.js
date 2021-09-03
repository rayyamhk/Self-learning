const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const OptimizeCssAssetsPlugin = require("optimize-css-assets-webpack-plugin");
const TerserPlugin = require("terser-webpack-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
    mode: "production",
    entry: "./src/js/app.js",
    output: {
        path: __dirname + "/dist",
        filename: "[contenthash].bundle.js" //common filename will cause browser caching
    },
    module: {
        rules: [
            {
                //Load css content into js, then extract it out into one css file
                //better user experience
                test: /\.css$/,
                use: [MiniCssExtractPlugin.loader, "css-loader"]
            },
            {
                test: /\.html$/,
                use: "html-loader"
            },
            {
                test: /\.(jpg|jpeg|png|svg|gif)$/,
                use: {
                    loader: "file-loader",
                    options: {
                        name: "[name].[hash].[ext]",
                        outputPath: "img",
                        esModule: false
                    }
                }
            }
        ]
    },
    plugins: [
        //refers to the loaders above
        new MiniCssExtractPlugin({
            filename: "[hash].css"
        })
    ],
    optimization: {
        //by default it is TerserPlugin, which is used to minify js file,
        //now you override it, you should add it back manually
        minimizer: [
            //minify the external css file generated by MiniCssExtractPlugin
            new OptimizeCssAssetsPlugin(), 
            new TerserPlugin(),
            //If you don't need to minify the html, then add it into plugins
            //Note that all js and css name are dynamic, so we need to add 
            //them dynamically, and this plugin provides the functionality
            new HtmlWebpackPlugin({
                filename: "[hash].html",
                template: "./src/index.html",
                minify: {
                    removeAttributeQuotes: true,
                    collapseWhitespace: true,
                    removeComments: true
                }
            })
        ]
    }
};
