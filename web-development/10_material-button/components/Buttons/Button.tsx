import React from "react";
import styled, { ThemeProvider} from "styled-components";
import Ripple from "../StyledComponents/Ripple";
import { RippleTrigger, BoxShadowStyle, getRGBA } from "../../functions"; 

interface Props {
    text: string;
    fontSize?: string;
    color?: string;
    backgroundColor?: string;
    opacity?: string;
    padding?: string;
    margin?: string;
    border?: string;
    rippleColor?: string;
}

const buttonTheme = {
    small: "0.5rem 1rem",
    regular: "0.75rem 1.75rem",
    large: "1.125rem 2.25rem",
}

const Button:React.FC<Props> = ({ text, fontSize="0.75rem", color = "#ffffff", backgroundColor = "#9c27b0", opacity = "1", padding = "0.75rem 1.75rem", margin = "1rem", border = "2px", rippleColor = "#ffffff" }) => {
    const rippleRef = React.createRef() as React.RefObject<HTMLSpanElement>;
      
    return (
        <ThemeProvider theme={buttonTheme}>
            <ButtonContainer type="button" fontsize={fontSize} color={color} background={backgroundColor} opacity={opacity} padding={padding} margin={margin} border={border} onClick={e => RippleTrigger(e, rippleRef.current as HTMLSpanElement)}>
                {text}
                <Ripple ref={rippleRef} color={rippleColor}/>
            </ButtonContainer>
        </ThemeProvider>
    ) 
}

export default Button;

interface ButtonProps {
    fontsize: string;
    color: string;
    background: string;
    opacity: string;
    padding: string;
    margin: string;
    border: string;
}
const ButtonContainer = styled.button<ButtonProps>`
    color: ${props => props.color};
    font-size: ${props => props.fontsize};
    background-color: ${props => getRGBA(props.background, props.opacity as any)};
    box-shadow: ${props => BoxShadowStyle(false, props.background)};
    border-radius: ${props => props.border};
    border: none;  
    outline: none; 
    padding: ${props => props.padding};
    margin: ${props => props.margin};
    display: inline-block;
    position: relative;
    transition: ${props => props.theme.primaryTransition};
    cursor: pointer;
    overflow: hidden;

    &:hover,
    &:focus {
        box-shadow: ${props => BoxShadowStyle(true, props.background)};
    }
`