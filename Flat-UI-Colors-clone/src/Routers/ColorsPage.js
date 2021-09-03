import React, {useState} from "react";
import ColorElement from "../Components/ColorElement";
import TopBar from "../Components/TopBar";
import Overlay from "../Components/Overlay";
import SEO from "../Components/SEO";
import ColorSet from "../Data/ColorSet";
import audioURL from "../Media/click.mp3";

function ColorsPage({match}) {
    const  [colorFormat, setColorFormat] = useState("hex1");
    const [color, setColor] = useState("");
    const [sound, setSound] = useState(true);
    const audio = new Audio(audioURL);
    const id = match.params.id;
    const colors = getColorsByID(ColorSet, id);

    const selectColor = color => {
        setColor(color);
        if(sound) {
            audio.play();
        }
    }

    const selectColorFormat = (e) => {
        setColorFormat(e.target.textContent.split(" ")[0]);
        document.querySelector(".copy-format").textContent = e.target.textContent;
    }

    const soundControl = () => {
        setSound(!sound);
    }

    return(
        <>
            <SEO title={colors.title + " | Flat UI Colors"} />
            <div className="colors-block-container">
                <TopBar 
                    sound={sound} 
                    selectColorFormat={selectColorFormat} 
                    soundControl={soundControl}
                />
                <div className="colors-block">
                    {
                        colors.colorSet.map((colorInfo, index) => 
                            <ColorElement 
                                color={colorFormat==="hex1"?colorInfo.hex1
                                        :colorFormat==="hex2"?colorInfo.hex2
                                        :colorFormat==="rgb"?colorInfo.rgb
                                        :colorFormat==="rgba"?colorInfo.rgba
                                        :""
                                    }
                                colorFormat={colorFormat}
                                text={colorInfo.text} 
                                selectColor={selectColor}
                                key={index} />
                        )
                    }
                </div>
            </div>
            <Overlay color={color} colorFormat={colorFormat}/>
        </>
    )
}

function getColorsByID(colorSet, id) {
    for(let i = 0; i < colorSet.length; i++) {
        if(colorSet[i].id === id) {
            return colorSet[i];
        }
    }
    return null;
}

export default ColorsPage;