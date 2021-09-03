# webpack-config-example
Webpack dev and prod config examples, all dependencies are installed as --save-dev

# npm i --save-dev webpack webpack-cli
where cli refers to command line interface, which allows you to use webpack with command line

# package.json
"start": "webpack-dev-server --config webpack.dev.js --open" <br/>
"build": "webpack --config webpack.prod.js" <br/>
webpack-dev-server allows you to host the website in memory without files, and it allows you to auto-reload
