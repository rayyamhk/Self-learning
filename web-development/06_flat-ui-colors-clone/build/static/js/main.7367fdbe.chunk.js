(this["webpackJsonpcolor-picker"]=this["webpackJsonpcolor-picker"]||[]).push([[0],{24:function(e,r,a){e.exports=a.p+"static/media/click.e3789c2b.mp3"},27:function(e,r,a){e.exports=a(41)},40:function(e,r,a){},41:function(e,r,a){"use strict";a.r(r);var t=a(0),b=a.n(t),g=a(17),x=a.n(g),n=a(18),c=a(19),h=a(25),l=a(20),o=a(26),f=a(8),i=a(5),d=a(9);var s=function(e){var r=e.color,a=e.colorFormat,t=e.text,g=e.selectColor,x={backgroundColor:""};return x.backgroundColor="hex2"===a?"#"+r:r,b.a.createElement("div",{className:"color-element",style:x,onClick:function(){!function(e){var r=document.createElement("textarea");r.value=e,r.setAttribute("readonly",""),r.style.position="absolute",r.style.left="-9999px",document.body.appendChild(r);var a=document.getSelection().rangeCount>0&&document.getSelection().getRangeAt(0);r.select(),document.execCommand("copy"),document.body.removeChild(r),a&&(document.getSelection().removeAllRanges(),document.getSelection().addRange(a))}(r),g(r),document.querySelector(".overlay").style.zIndex="1",document.querySelector(".overlay").style.opacity="1",setTimeout((function(){document.querySelector(".overlay").style.zIndex="-1",document.querySelector(".overlay").style.opacity="0"}),750)}},b.a.createElement("button",null,"copy"),b.a.createElement("span",null,t))};var u=function(e){var r=e.sound,a=e.selectColorFormat,g=e.soundControl,x=Object(t.useState)(!1),n=Object(d.a)(x,2),c=n[0],h=n[1];return b.a.createElement("div",{className:"top-bar"},b.a.createElement(f.b,{to:"/",className:"go-back"},"back"),b.a.createElement("div",{onClick:function(){h(!c)}},b.a.createElement("span",null,"copy format: "),b.a.createElement("span",{className:"copy-format"},"hex1 (#aa1923)"),b.a.createElement("div",{className:c?"format-list":"format-list-closed"},b.a.createElement("span",{onClick:function(e){return a(e)}},"hex1 (#aa1923)"),b.a.createElement("span",{onClick:function(e){return a(e)}},"hex2 (aa1923)"),b.a.createElement("span",{onClick:function(e){return a(e)}},"rgb - (1,2,3)"),b.a.createElement("span",{onClick:function(e){return a(e)}},"rgba - (1,2,3,0.4)"))),b.a.createElement("span",{className:"sound-control",onClick:function(){return g()}},r?"sound on":"sound off"))};var m=function(e){var r=e.color,a=e.colorFormat,t={backgroundColor:""};return t.backgroundColor="hex2"===a?"#"+r:r,b.a.createElement("div",{className:"overlay",style:t},b.a.createElement("div",null,b.a.createElement("span",{className:"overlay-popup"},"copied"),b.a.createElement("span",{className:"overlay-color"},r)))},p=a(23);var y=function(e){var r=e.title;return b.a.createElement(p.Helmet,null,b.a.createElement("title",null,r))},v=[{title:"Flat UI Palette",id:"default",colorSet:[{hex1:"#1abc9c",hex2:"1abc9c",rgb:"rgb(26, 188, 156)",rgba:"rgba(26, 188, 156,1.0)",text:"turquoise"},{hex1:"#2ecc71",hex2:"2ecc71",rgb:"rgb(46, 204, 113)",rgba:"rgba(46, 204, 113,1.0)",text:"emerald"},{hex1:"#3498db",hex2:"3498db",rgb:"rgb(52, 152, 219)",rgba:"rgba(52, 152, 219,1.0)",text:"peter river"},{hex1:"#9b59b6",hex2:"9b59b6",rgb:"rgb(155, 89, 182)",rgba:"rgba(155, 89, 182,1.0)",text:"amethyst"},{hex1:"#34495e",hex2:"34495e",rgb:"rgb(52, 73, 94)",rgba:"rgba(52, 73, 94,1.0)",text:"wet asphalt"},{hex1:"#16a085",hex2:"16a085",rgb:"rgb(22, 160, 133)",rgba:"rgba(22, 160, 133,1.0)",text:"green sea"},{hex1:"#27ae60",hex2:"27ae60",rgb:"rgb(39, 174, 96)",rgba:"rgba(39, 174, 96,1.0)",text:"nephritis"},{hex1:"#2980b9",hex2:"2980b9",rgb:"rgb(41, 128, 185)",rgba:"rgba(41, 128, 185,1.0)",text:"belize hole"},{hex1:"#8e44ad",hex2:"8e44ad",rgb:"rgb(142, 68, 173)",rgba:"rgba(142, 68, 173,1.0)",text:"wisteria"},{hex1:"#2c3e50",hex2:"2c3e50",rgb:"rgb(44, 62, 80)",rgba:"rgba(44, 62, 80,1.0)",text:"midnight blue"},{hex1:"#f1c40f",hex2:"f1c40f",rgb:"rgb(241, 196, 15)",rgba:"rgba(241, 196, 15,1.0)",text:"sun flower"},{hex1:"#e67e22",hex2:"e67e22",rgb:"rgb(230, 126, 34)",rgba:"rgba(230, 126, 34,1.0)",text:"carrot"},{hex1:"#e74c3c",hex2:"e74c3c",rgb:"rgb(231, 76, 60)",rgba:"rgba(231, 76, 60,1.0)",text:"alizarin"},{hex1:"#ecf0f1",hex2:"ecf0f1",rgb:"rgb(236, 240, 241)",rgba:"rgba(236, 240, 241,1.0)",text:"clouds"},{hex1:"#95a5a6",hex2:"95a5a6",rgb:"rgb(149, 165, 166)",rgba:"rgba(149, 165, 166,1.0)",text:"concrete"},{hex1:"#f39c12",hex2:"f39c12",rgb:"rgb(243, 156, 18)",rgba:"rgba(243, 156, 18,1.0)",text:"orange"},{hex1:"#d35400",hex2:"d35400",rgb:"rgb(211, 84, 0)",rgba:"rgba(211, 84, 0,1.0)",text:"pumpkin"},{hex1:"#c0392b",hex2:"c0392b",rgb:"rgb(192, 57, 43)",rgba:"rgba(192, 57, 43,1.0)",text:"pomegranate"},{hex1:"#bdc3c7",hex2:"bdc3c7",rgb:"rgb(189, 195, 199)",rgba:"rgba(189, 195, 199,1.0)",text:"silver"},{hex1:"#7f8c8d",hex2:"7f8c8d",rgb:"rgb(127, 140, 141)",rgba:"rgba(127, 140, 141,1.0)",text:"asbestos"}]},{title:"British Palette",id:"uk",colorSet:[{hex1:"#00a8ff",hex2:"00a8ff",rgb:"rgb(0, 168, 255)",rgba:"rgba(0, 168, 255,1.0)",text:"protoss pylo"},{hex1:"#9c88ff",hex2:"9c88ff",rgb:"rgb(156, 136, 255)",rgba:"rgba(156, 136, 255,1.0)",text:"periwinkle"},{hex1:"#fbc531",hex2:"fbc531",rgb:"rgb(251, 197, 49)",rgba:"rgba(251, 197, 49,1.0)",text:"rise-n-shine"},{hex1:"#4cd137",hex2:"4cd137",rgb:"rgb(76, 209, 55)",rgba:"rgba(76, 209, 55,1.0)",text:"download progress"},{hex1:"#487eb0",hex2:"487eb0",rgb:"rgb(72, 126, 176)",rgba:"rgba(72, 126, 176,1.0)",text:"seabrook"},{hex1:"#0097e6",hex2:"0097e6",rgb:"rgb(0, 151, 230)",rgba:"rgba(0, 151, 230,1.0)",text:"vanadyl blue"},{hex1:"#8c7ae6",hex2:"8c7ae6",rgb:"rgb(140, 122, 230)",rgba:"rgba(140, 122, 230,1.0)",text:"matt purple"},{hex1:"#e1b12c",hex2:"e1b12c",rgb:"rgb(225, 177, 44)",rgba:"rgba(225, 177, 44,1.0)",text:"nanohanacha gold"},{hex1:"#44bd32",hex2:"44bd32",rgb:"rgb(68, 189, 50)",rgba:"rgba(68, 189, 50,1.0)",text:"skirret green"},{hex1:"#40739e",hex2:"40739e",rgb:"rgb(64, 115, 158)",rgba:"rgba(64, 115, 158,1.0)",text:"naval"},{hex1:"#e84118",hex2:"e84118",rgb:"rgb(232, 65, 24)",rgba:"rgba(232, 65, 24,1.0)",text:"nasturcian flower"},{hex1:"#f5f6fa",hex2:"f5f6fa",rgb:"rgb(245, 246, 250)",rgba:"rgba(245, 246, 250,1.0)",text:"lynx white"},{hex1:"#7f8fa6",hex2:"7f8fa6",rgb:"rgb(127, 143, 166)",rgba:"rgba(127, 143, 166,1.0)",text:"blueberry soda"},{hex1:"#273c75",hex2:"273c75",rgb:"rgb(39, 60, 117)",rgba:"rgba(39, 60, 117,1.0)",text:"mazarine blue"},{hex1:"#353b48",hex2:"353b48",rgb:"rgb(53, 59, 72)",rgba:"rgba(53, 59, 72,1.0)",text:"blue nights"},{hex1:"#c23616",hex2:"c23616",rgb:"rgb(194, 54, 22)",rgba:"rgba(194, 54, 22,1.0)",text:"harley davidson orange"},{hex1:"#dcdde1",hex2:"dcdde1",rgb:"rgb(220, 221, 225)",rgba:"rgba(220, 221, 225,1.0)",text:"hint of pensive"},{hex1:"#718093",hex2:"718093",rgb:"rgb(113, 128, 147)",rgba:"rgba(113, 128, 147,1.0)",text:"chain gang grey"},{hex1:"#192a56",hex2:"192a56",rgb:"rgb(25, 42, 86)",rgba:"rgba(25, 42, 86,1.0)",text:"pico void"},{hex1:"#2f3640",hex2:"2f3640",rgb:"rgb(47, 54, 64)",rgba:"rgba(47, 54, 64,1.0)",text:"electromagnnetic"}]},{title:"French Palette",id:"fr",colorSet:[{hex1:"#fad390",hex2:"fad390",rgb:"rgb(250, 211, 144)",rgba:"rgba(250, 211, 144,1.0)",text:"flat flesh"},{hex1:"#f8c291",hex2:"f8c291",rgb:"rgb(248, 194, 145)",rgba:"rgba(248, 194, 145,1.0)",text:"melon melody"},{hex1:"#6a89cc",hex2:"6a89cc",rgb:"rgb(106, 137, 204)",rgba:"rgba(106, 137, 204,1.0)",text:"livid"},{hex1:"#82ccdd",hex2:"82ccdd",rgb:"rgb(130, 204, 221)",rgba:"rgba(130, 204, 221,1.0)",text:"spray"},{hex1:"#b8e994",hex2:"b8e994",rgb:"rgb(184, 233, 148)",rgba:"rgba(184, 233, 148,1.0)",text:"paradise green"},{hex1:"#f6b93b",hex2:"f6b93b",rgb:"rgb(246, 185, 59)",rgba:"rgba(246, 185, 59,1.0)",text:"squash blossom"},{hex1:"#e55039",hex2:"e55039",rgb:"rgb(229, 80, 57)",rgba:"rgba(229, 80, 57,1.0)",text:"mandarin red"},{hex1:"#4a69bd",hex2:"4a69bd",rgb:"rgb(74, 105, 189)",rgba:"rgba(74, 105, 189,1.0)",text:"azraq blue"},{hex1:"#60a3bc",hex2:"60a3bc",rgb:"rgb(96, 163, 188)",rgba:"rgba(96, 163, 188,1.0)",text:"dupain"},{hex1:"#78e08f",hex2:"78e08f",rgb:"rgb(120, 224, 143)",rgba:"rgba(120, 224, 143,1.0)",text:"aurora green"},{hex1:"#fa983a",hex2:"fa983a",rgb:"rgb(250, 152, 58)",rgba:"rgba(250, 152, 58,1.0)",text:"iceland poppy"},{hex1:"#eb2f06",hex2:"eb2f06",rgb:"rgb(235, 47, 6)",rgba:"rgba(235, 47, 6,1.0)",text:"tomato red"},{hex1:"#1e3799",hex2:"1e3799",rgb:"rgb(30, 55, 153)",rgba:"rgba(30, 55, 153,1.0)",text:"yue guang lan blue"},{hex1:"#3c6382",hex2:"3c6382",rgb:"rgb(60, 99, 130)",rgba:"rgba(60, 99, 130,1.0)",text:"good samaritan"},{hex1:"#38ada9",hex2:"38ada9",rgb:"rgb(56, 173, 169)",rgba:"rgba(56, 173, 169,1.0)",text:"waterfall"},{hex1:"#e58e26",hex2:"e58e26",rgb:"rgb(229, 142, 38)",rgba:"rgba(229, 142, 38,1.0)",text:"carrot orange"},{hex1:"#b71540",hex2:"b71540",rgb:"rgb(183, 21, 64)",rgba:"rgba(183, 21, 64,1.0)",text:"jalapeno red"},{hex1:"#0c2461",hex2:"0c2461",rgb:"rgb(12, 36, 97)",rgba:"rgba(12, 36, 97, 1.0)",text:"dark sapphire"},{hex1:"#0a3d62",hex2:"0a3d62",rgb:"rgb(10, 61, 98)",rgba:"rgba(10, 61, 98,1.0)",text:"forest blues"},{hex1:"#079992",hex2:"079992",rgb:"rgb(7, 153, 146)",rgba:"rgba(7, 153, 146, 1.0)",text:"reef encounter"}]},{title:"Canadian Palette",id:"ca",colorSet:[{hex1:"#ff9ff3",hex2:"ff9ff3",rgb:"rgb(255, 159, 243)",rgba:"rgba(255, 159, 243,1.0)",text:"jigglypuff"},{hex1:"#feca57",hex2:"feca57",rgb:"rgb(254, 202, 87)",rgba:"rgba(254, 202, 87,1.0)",text:"casandora yellow"},{hex1:"#ff6b6b",hex2:"ff6b6b",rgb:"rgb(255, 107, 107)",rgba:"rgba(255, 107, 107,1.0)",text:"pastel red"},{hex1:"#48dbfb",hex2:"48dbfb",rgb:"rgb(72, 219, 251)",rgba:"rgba(72, 219, 251,1.0)",text:"megaman"},{hex1:"#1dd1a1",hex2:"1dd1a1",rgb:"rgb(29, 209, 161)",rgba:"rgba(29, 209, 161,1.0)",text:"wild caribbean green"},{hex1:"#f368e0",hex2:"f368e0",rgb:"rgb(243, 104, 224)",rgba:"rgba(243, 104, 224,1.0)",text:"lian hong lotus pink"},{hex1:"#ff9f43",hex2:"ff9f43",rgb:"rgb(255, 159, 67)",rgba:"rgba(255, 159, 67,1.0)",text:"double dragon skin"},{hex1:"#ee5253",hex2:"ee5253",rgb:"rgb(238, 82, 83)",rgba:"rgba(238, 82, 83,1.0)",text:"amour"},{hex1:"#0abde3",hex2:"0abde3",rgb:"rgb(10, 189, 227)",rgba:"rgba(10, 189, 227,1.0)",text:"cyanite"},{hex1:"#10ac84",hex2:"10ac84",rgb:"rgb(16, 172, 132)",rgba:"rgba(16, 172, 132,1.0)",text:"dark mountain meadow"},{hex1:"#00d2d3",hex2:"00d2d3",rgb:"rgb(0, 210, 211)",rgba:"rgba(0, 210, 211,1.0)",text:"jade dust"},{hex1:"#54a0ff",hex2:"54a0ff",rgb:"rgb(84, 160, 255)",rgba:"rgba(84, 160, 255,1.0)",text:"joust blue"},{hex1:"#5f27cd",hex2:"5f27cd",rgb:"rgb(95, 39, 205)",rgba:"rgba(95, 39, 205,1.0)",text:"nasu purple"},{hex1:"#c8d6e5",hex2:"c8d6e5",rgb:"rgb(200, 214, 229)",rgba:"rgba(200, 214, 229,1.0)",text:"light blue ballerina"},{hex1:"#576574",hex2:"576574",rgb:"rgb(87, 101, 116)",rgba:"rgba(87, 101, 116,1.0)",text:"fuel town"},{hex1:"#01a3a4",hex2:"01a3a4",rgb:"rgb(1, 163, 164)",rgba:"rgba(1, 163, 164,1.0)",text:"aqua velvet"},{hex1:"#2e86de",hex2:"2e86de",rgb:"rgb(46, 134, 222)",rgba:"rgba(46, 134, 222,1.0)",text:"blue de france"},{hex1:"#341f97",hex2:"341f97",rgb:"rgb(52, 31, 151)",rgba:"rgba(52, 31, 151,1.0)",text:"bluebell"},{hex1:"#8395a7",hex2:"8395a7",rgb:"rgb(131, 149, 167)",rgba:"rgba(131, 149, 167,1.0)",text:"storm petrel"},{hex1:"#222f3e",hex2:"222f3e",rgb:"rgb(34, 47, 62)",rgba:"rgba(34, 47, 62,1.0)",text:"imperial primer"}]},{title:"Chinese Palette",id:"cn",colorSet:[{hex1:"#eccc68",hex2:"eccc68",rgb:"rgb(236, 204, 104)",rgba:"rgba(236, 204, 104,1.0)",text:"golden sand"},{hex1:"#ff7f50",hex2:"ff7f50",rgb:"rgb(255, 127, 80)",rgba:"rgba(255, 127, 80,1.0)",text:"coral"},{hex1:"#ff6b81",hex2:"ff6b81",rgb:"rgb(255, 107, 129)",rgba:"rgb(255, 107, 129,1.0)",text:"wild watermelon"},{hex1:"#a4b0be",hex2:"a4b0be",rgb:"rgb(164, 176, 190)",rgba:"rgba(164, 176, 190,1.0)",text:"peace"},{hex1:"#57606f",hex2:"57606f",rgb:"rgb(87, 96, 111)",rgba:"rgba(87, 96, 111,1.0)",text:"grisaille"},{hex1:"#ffa502",hex2:"ffa502",rgb:"rgb(255, 165, 2)",rgba:"rgba(255, 165, 2,1.0)",text:"orange"},{hex1:"#ff6348",hex2:"ff6348",rgb:"rgb(255, 99, 72)",rgba:"rgba(255, 99, 72,1.0)",text:"bruschetta tomato"},{hex1:"#ff4757",hex2:"ff4757",rgb:"rgb(255, 71, 87)",rgba:"rgba(255, 71, 87,1.0)",text:"watermelon"},{hex1:"#747d8c",hex2:"747d8c",rgb:"rgb(116, 125, 140)",rgba:"rgba(116, 125, 140,1.0)",text:"bay wharf"},{hex1:"#2f3542",hex2:"2f3542",rgb:"rgb(47, 53, 66)",rgba:"rgba(47, 53, 66,1.0)",text:"prestige blue"},{hex1:"#7bed9f",hex2:"7bed9f",rgb:"rgb(123, 237, 159)",rgba:"rgba(123, 237, 159,1.0)",text:"lime soap"},{hex1:"#70a1ff",hex2:"70a1ff",rgb:"rgb(112, 161, 255)",rgba:"rgba(112, 161, 255,1.0)",text:"french sky blue"},{hex1:"#5352ed",hex2:"5352ed",rgb:"rgb(83, 82, 237)",rgba:"rgba(83, 82, 237,1.0)",text:"saturated sky"},{hex1:"#ffffff",hex2:"ffffff",rgb:"rgb(255, 255, 255)",rgba:"rgba(255, 255, 255,1.0)",text:"white"},{hex1:"#dfe4ea",hex2:"dfe4ea",rgb:"rgb(223, 228, 234)",rgba:"rgba(223, 228, 234,1.0)",text:"city lights"},{hex1:"#2ed573",hex2:"2ed573",rgb:"rgb(46, 213, 115)",rgba:"rgba(46, 213, 115,1.0)",text:"ufo green"},{hex1:"#1e90ff",hex2:"1e90ff",rgb:"rgb(30, 144, 255)",rgba:"rgba(30, 144, 255,1.0)",text:"clear chill"},{hex1:"#3742fa",hex2:"3742fa",rgb:"rgb(55, 66, 250)",rgba:"rgba(55, 66, 250,1.0)",text:"bright greek"},{hex1:"#f1f2f6",hex2:"f1f2f6",rgb:"rgb(241, 242, 246)",rgba:"rgba(241, 242, 246,1.0)",text:"anti-flash white"},{hex1:"#ced6e0",hex2:"ced6e0",rgb:"rgb(206, 214, 224)",rgba:"rgba(206, 214, 224,1.0)",text:"twinkle blue"}]},{title:"Swedish Palette",id:"se",colorSet:[{hex1:"#ef5777",hex2:"ef5777",rgb:"rgb(239, 87, 119)",rgba:"rgba(239, 87, 119,1.0)",text:"highlighter pink"},{hex1:"#575fcf",hex2:"575fcf",rgb:"rgb(60, 64, 198)",rgba:"rgba(60, 64, 198,1.0)",text:"dark periwinkle"},{hex1:"#4bcffa",hex2:"4bcffa",rgb:"rgb(75, 207, 250)",rgba:"rgba(75, 207, 250,1.0)",text:"megaman"},{hex1:"#34e7e4",hex2:"34e7e4",rgb:"rgb(52, 231, 228)",rgba:"rgba(52, 231, 228,1.0)",text:"fresh turquoise"},{hex1:"#0be881",hex2:"0be881",rgb:"rgb(11, 232, 129)",rgba:"rgba(11, 232, 129,1.0)",text:"minty green"},{hex1:"#f53b57",hex2:"f53b57",rgb:"rgb(245, 59, 87)",rgba:"rgba(245, 59, 87,1.0)",text:"sizzling red"},{hex1:"#3c40c6",hex2:"3c40c6",rgb:"rgb(60, 64, 198)",rgba:"rgba(60, 64, 198,1.0)",text:"free speech blue"},{hex1:"#0fbcf9",hex2:"0fbcf9",rgb:"rgb(15, 188, 249)",rgba:"rgba(15, 188, 249,1.0)",text:"spiro disco ball"},{hex1:"#00d8d6",hex2:"00d8d6",rgb:"rgb(0, 216, 214)",rgba:"rgba(0, 216, 214,1.0)",text:"jade dust"},{hex1:"#05c46b",hex2:"05c46b",rgb:"rgb(5, 196, 107)",rgba:"rgba(5, 196, 107,1.0)",text:"green teal"},{hex1:"#ffc048",hex2:"ffc048",rgb:"rgb(255, 192, 72)",rgba:"rgba(255, 192, 72,1.0)",text:"narenji orange"},{hex1:"#ffdd59",hex2:"ffdd59",rgb:"rgb(255, 221, 89)",rgba:"rgba(255, 221, 89,1.0)",text:"yriel yellow"},{hex1:"#ff5e57",hex2:"ff5e57",rgb:"rgb(255, 94, 87)",rgba:"rgba(255, 94, 87,1.0)",text:"sunset orange"},{hex1:"#d2dae2",hex2:"d2dae2",rgb:"rgb(210, 218, 226)",rgba:"rgba(210, 218, 226,1.0)",text:"hint of elusive blue"},{hex1:"#485460",hex2:"485460",rgb:"rgb(72, 84, 96)",rgba:"rgba(72, 84, 96,1.0)",text:"good night!"},{hex1:"#ffa801",hex2:"ffa801",rgb:"rgb(255, 168, 1)",rgba:"rgba(255, 168, 1,1.0)",text:"chrome yellow"},{hex1:"#ffd32a",hex2:"ffd32a",rgb:"rgb(255, 211, 42)",rgba:"rgba(255, 211, 42,1.0)",text:"vibrant yellow"},{hex1:"#ff3f34",hex2:"ff3f34",rgb:"rgb(255, 63, 52)",rgba:"rgba(255, 63, 52,1.0)",text:"red orange"},{hex1:"#808e9b",hex2:"808e9b",rgb:"rgb(128, 142, 155)",rgba:"rgba(128, 142, 155,1.0)",text:"london square"},{hex1:"#1e272e",hex2:"1e272e",rgb:"rgb(30, 39, 46)",rgba:"rgba(30, 39, 46,1.0)",text:"black pearl"}]}],k=a(24),E=a.n(k);var C=function(e){var r=e.match,a=Object(t.useState)("hex1"),g=Object(d.a)(a,2),x=g[0],n=g[1],c=Object(t.useState)(""),h=Object(d.a)(c,2),l=h[0],o=h[1],f=Object(t.useState)(!0),i=Object(d.a)(f,2),p=i[0],k=i[1],C=new Audio(E.a),w=r.params.id,S=function(e,r){for(var a=0;a<e.length;a++)if(e[a].id===r)return e[a];return null}(v,w),j=function(e){o(e),p&&C.play()};return b.a.createElement(b.a.Fragment,null,b.a.createElement(y,{title:S.title+" | Flat UI Colors"}),b.a.createElement("div",{className:"colors-block-container"},b.a.createElement(u,{sound:p,selectColorFormat:function(e){n(e.target.textContent.split(" ")[0]),document.querySelector(".copy-format").textContent=e.target.textContent},soundControl:function(){k(!p)}}),b.a.createElement("div",{className:"colors-block"},S.colorSet.map((function(e,r){return b.a.createElement(s,{color:"hex1"===x?e.hex1:"hex2"===x?e.hex2:"rgb"===x?e.rgb:"rgba"===x?e.rgba:"",colorFormat:x,text:e.text,selectColor:j,key:r})})))),b.a.createElement(m,{color:l,colorFormat:x}))};var w=function(e){var r=e.colors;return b.a.createElement(f.b,{to:"/palette/"+r.id,className:"colors-box-anchor"},b.a.createElement("div",{className:"colors-box"},b.a.createElement("div",null,r.colorSet.map((function(e,r){return b.a.createElement("span",{style:{backgroundColor:e.hex1,minHeight:"40px"},key:r})}))),b.a.createElement("span",null,r.title)))};var S=function(){return b.a.createElement("div",{className:"container"},b.a.createElement(y,{title:"Palettes | Flat UI Colors"}),v.map((function(e,r){return b.a.createElement(w,{colors:e,key:r})})))},j=function(e){function r(){return Object(n.a)(this,r),Object(h.a)(this,Object(l.a)(r).apply(this,arguments))}return Object(o.a)(r,e),Object(c.a)(r,[{key:"render",value:function(){return b.a.createElement(f.a,{basename:"https://rayyamhk.github.io/Self-learning/web-development/06_flat-ui-colors-clone/build"},b.a.createElement(i.c,null,b.a.createElement(i.a,{exact:!0,path:"/",component:S}),b.a.createElement(i.a,{exact:!0,path:"/palette/:id",component:C})))}}]),r}(b.a.Component);a(40);x.a.render(b.a.createElement(j,null),document.getElementById("root"))}},[[27,1,2]]]);
//# sourceMappingURL=main.7367fdbe.chunk.js.map