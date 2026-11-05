import './ui/style.css';
import { App, colorVars } from './ui/app';

const root = document.getElementById('app')!;
root.setAttribute('style', colorVars());
new App(root);
