---
title: "SonicMoE × Blackwell 深度解读：fine-grained MoE kernel 的算法 × 软件 × 硬件三层堆叠"
date: 2026-04-27T19:18:52+08:00
draft: false
tags: ["sonicmoe", "moe", "blackwell", "cuda", "gpu-kernel", "个人笔记"]
---

<style>#back-to-top{background:#000;-webkit-border-radius:50%;-moz-border-radius:50%;border-radius:50%;bottom:20px;-webkit-box-shadow:0 2px 5px 0 rgba(0,0,0,.26);-moz-box-shadow:0 2px 5px 0 rgba(0,0,0,.26);box-shadow:0 2px 5px 0 rgba(0,0,0,.26);color:#fff;cursor:pointer;display:block;height:56px;opacity:1;outline:0;position:fixed;right:20px;-webkit-tap-highlight-color:transparent;-webkit-touch-callout:none;-webkit-transition:bottom .2s,opacity .2s;-o-transition:bottom .2s,opacity .2s;-moz-transition:bottom .2s,opacity .2s;transition:bottom .2s,opacity .2s;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;width:56px;z-index:1}#back-to-top svg{display:block;fill:currentColor;height:20px;margin:11px auto 0;width:20px}#back-to-top.hidden{bottom:-56px;opacity:0}</style>
<style id="distill-prerendered-styles" type="text/css">/*
 * Copyright 2018 The Distill Template Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

html {
  font-size: 14px;
	line-height: 1.6em;
  /* font-family: "Libre Franklin", "Helvetica Neue", sans-serif; */
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", Arial, sans-serif;
  /*, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";*/
  text-size-adjust: 100%;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}

@media(min-width: 768px) {
  html {
    font-size: 16px;
  }
}

body {
  margin: 0;
}

a {
  color: #004276;
}

figure {
  margin: 0;
}

table {
	border-collapse: collapse;
	border-spacing: 0;
}

table th {
	text-align: left;
}

table thead {
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

table thead th {
  padding-bottom: 0.5em;
}

table tbody :first-child td {
  padding-top: 0.5em;
}

pre {
  overflow: auto;
  max-width: 100%;
}

p {
  margin-top: 0;
  margin-bottom: 1em;
}

sup, sub {
  vertical-align: baseline;
  position: relative;
  top: -0.4em;
  line-height: 1em;
}

sub {
  top: 0.4em;
}

.kicker,
.marker {
  font-size: 15px;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.5);
}


/* Headline */

@media(min-width: 1024px) {
  d-title h1 span {
    display: block;
  }
}

/* Figure */

figure {
  position: relative;
  margin-bottom: 2.5em;
  margin-top: 1.5em;
}

figcaption+figure {

}

figure img {
  width: 100%;
}

figure svg text,
figure svg tspan {
}

figcaption,
.figcaption {
  color: rgba(0, 0, 0, 0.6);
  font-size: 12px;
  line-height: 1.5em;
}

@media(min-width: 1024px) {
figcaption,
.figcaption {
    font-size: 13px;
  }
}

figure.external img {
  background: white;
  border: 1px solid rgba(0, 0, 0, 0.1);
  box-shadow: 0 1px 8px rgba(0, 0, 0, 0.1);
  padding: 18px;
  box-sizing: border-box;
}

figcaption a {
  color: rgba(0, 0, 0, 0.6);
}

figcaption b,
figcaption strong, {
  font-weight: 600;
  color: rgba(0, 0, 0, 1.0);
}
/*
 * Copyright 2018 The Distill Template Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

@supports not (display: grid) {
  .base-grid,
  distill-header,
  d-title,
  d-abstract,
  d-article,
  d-appendix,
  distill-appendix,
  d-byline,
  d-footnote-list,
  d-citation-list,
  distill-footer {
    display: block;
    padding: 8px;
  }
}

.base-grid,
distill-header,
d-title,
d-abstract,
d-article,
d-appendix,
distill-appendix,
d-byline,
d-footnote-list,
d-citation-list,
distill-footer {
  display: grid;
  justify-items: stretch;
  grid-template-columns: [screen-start] 8px [page-start kicker-start text-start gutter-start middle-start] 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr [text-end page-end gutter-end kicker-end middle-end] 8px [screen-end];
  grid-column-gap: 8px;
}

.grid {
  display: grid;
  grid-column-gap: 8px;
}

@media(min-width: 768px) {
  .base-grid,
  distill-header,
  d-title,
  d-abstract,
  d-article,
  d-appendix,
  distill-appendix,
  d-byline,
  d-footnote-list,
  d-citation-list,
  distill-footer {
    grid-template-columns: [screen-start] 1fr [page-start kicker-start middle-start text-start] 45px 45px 45px 45px 45px 45px 45px 45px [ kicker-end text-end gutter-start] 45px [middle-end] 45px [page-end gutter-end] 1fr [screen-end];
    grid-column-gap: 16px;
  }

  .grid {
    grid-column-gap: 16px;
  }
}

@media(min-width: 1000px) {
  .base-grid,
  distill-header,
  d-title,
  d-abstract,
  d-article,
  d-appendix,
  distill-appendix,
  d-byline,
  d-footnote-list,
  d-citation-list,
  distill-footer {
    grid-template-columns: [screen-start] 1fr [page-start kicker-start] 50px [middle-start] 50px [text-start kicker-end] 50px 50px 50px 50px 50px 50px 50px 50px [text-end gutter-start] 50px [middle-end] 50px [page-end gutter-end] 1fr [screen-end];
    grid-column-gap: 16px;
  }

  .grid {
    grid-column-gap: 16px;
  }
}

@media(min-width: 1180px) {
  .base-grid,
  distill-header,
  d-title,
  d-abstract,
  d-article,
  d-appendix,
  distill-appendix,
  d-byline,
  d-footnote-list,
  d-citation-list,
  distill-footer {
    grid-template-columns: [screen-start] 1fr [page-start kicker-start] 60px [middle-start] 60px [text-start kicker-end] 60px 60px 60px 60px 60px 60px 60px 60px [text-end gutter-start] 60px [middle-end] 60px [page-end gutter-end] 1fr [screen-end];
    grid-column-gap: 32px;
  }

  .grid {
    grid-column-gap: 32px;
  }
}




.base-grid {
  grid-column: screen;
}

/* .l-body,
d-article > *  {
  grid-column: text;
}

.l-page,
d-title > *,
d-figure {
  grid-column: page;
} */

.l-gutter {
  grid-column: gutter;
}

.l-text,
.l-body {
  grid-column: text;
}

.l-page {
  grid-column: page;
}

.l-body-outset {
  grid-column: middle;
}

.l-page-outset {
  grid-column: page;
}

.l-screen {
  grid-column: screen;
}

.l-screen-inset {
  grid-column: screen;
  padding-left: 16px;
  padding-left: 16px;
}


/* Aside */

d-article aside {
  grid-column: gutter;
  font-size: 12px;
  line-height: 1.6em;
  color: rgba(0, 0, 0, 0.6)
}

@media(min-width: 768px) {
  aside {
    grid-column: gutter;
  }

  .side {
    grid-column: gutter;
  }
}
/*
 * Copyright 2018 The Distill Template Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

d-title {
  padding: 2rem 0 1.5rem;
  contain: layout style;
  overflow-x: hidden;
}

@media(min-width: 768px) {
  d-title {
    padding: 4rem 0 1.5rem;
  }
}

d-title h1 {
  grid-column: text;
  font-size: 40px;
  font-weight: 700;
  line-height: 1.1em;
  margin: 0 0 0.5rem;
}

@media(min-width: 768px) {
  d-title h1 {
    font-size: 50px;
  }
}

d-title p {
  font-weight: 300;
  font-size: 1.2rem;
  line-height: 1.55em;
  grid-column: text;
}

d-title .status {
  margin-top: 0px;
  font-size: 12px;
  color: #009688;
  opacity: 0.8;
  grid-column: kicker;
}

d-title .status span {
  line-height: 1;
  display: inline-block;
  padding: 6px 0;
  border-bottom: 1px solid #80cbc4;
  font-size: 11px;
  text-transform: uppercase;
}
/*
 * Copyright 2018 The Distill Template Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

d-byline {
  contain: style;
  overflow: hidden;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  font-size: 0.8rem;
  line-height: 1.8em;
  padding: 1.5rem 0;
  min-height: 1.8em;
}


d-byline .byline {
  grid-template-columns: 1fr 1fr;
  grid-column: text;
}

@media(min-width: 768px) {
  d-byline .byline {
    grid-template-columns: 1fr 1fr 1fr 1fr;
  }
}

d-byline .authors-affiliations {
  grid-column-end: span 2;
  grid-template-columns: 1fr 1fr;
  margin-bottom: 1em;
}

@media(min-width: 768px) {
  d-byline .authors-affiliations {
    margin-bottom: 0;
  }
}

d-byline h3 {
  font-size: 0.6rem;
  font-weight: 400;
  color: rgba(0, 0, 0, 0.5);
  margin: 0;
  text-transform: uppercase;
}

d-byline p {
  margin: 0;
}

d-byline a,
d-article d-byline a {
  color: rgba(0, 0, 0, 0.8);
  text-decoration: none;
  border-bottom: none;
}

d-article d-byline a:hover {
  text-decoration: underline;
  border-bottom: none;
}

d-byline p.author {
  font-weight: 500;
}

d-byline .affiliations {

}
/*
 * Copyright 2018 The Distill Template Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

d-article {
  contain: layout style;
  overflow-x: hidden;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  padding-top: 2rem;
  color: rgba(0, 0, 0, 0.8);
}

d-article > * {
  grid-column: text;
}

@media(min-width: 768px) {
  d-article {
    font-size: 16px;
  }
}

@media(min-width: 1024px) {
  d-article {
    font-size: 1.06rem;
    line-height: 1.7em;
  }
}


/* H2 */


d-article .marker {
  text-decoration: none;
  border: none;
  counter-reset: section;
  grid-column: kicker;
  line-height: 1.7em;
}

d-article .marker:hover {
  border: none;
}

d-article .marker span {
  padding: 0 3px 4px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.2);
  position: relative;
  top: 4px;
}

d-article .marker:hover span {
  color: rgba(0, 0, 0, 0.7);
  border-bottom: 1px solid rgba(0, 0, 0, 0.7);
}

d-article h2 {
  font-weight: 600;
  font-size: 24px;
  line-height: 1.25em;
  margin: 2rem 0 1.5rem 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  padding-bottom: 1rem;
}

@media(min-width: 1024px) {
  d-article h2 {
    font-size: 36px;
  }
}

/* H3 */

d-article h3 {
  font-weight: 700;
  font-size: 18px;
  line-height: 1.4em;
  margin-bottom: 1em;
  margin-top: 2em;
}

@media(min-width: 1024px) {
  d-article h3 {
    font-size: 20px;
  }
}

/* H4 */

d-article h4 {
  font-weight: 600;
  text-transform: uppercase;
  font-size: 14px;
  line-height: 1.4em;
}

d-article a {
  color: inherit;
}

d-article p,
d-article ul,
d-article ol,
d-article blockquote {
  margin-top: 0;
  margin-bottom: 1em;
  margin-left: 0;
  margin-right: 0;
}

d-article blockquote {
  border-left: 2px solid rgba(0, 0, 0, 0.2);
  padding-left: 2em;
  font-style: italic;
  color: rgba(0, 0, 0, 0.6);
}

d-article a {
  border-bottom: 1px solid rgba(0, 0, 0, 0.4);
  text-decoration: none;
}

d-article a:hover {
  border-bottom: 1px solid rgba(0, 0, 0, 0.8);
}

d-article .link {
  text-decoration: underline;
  cursor: pointer;
}

d-article ul,
d-article ol {
  padding-left: 24px;
}

d-article li {
  margin-bottom: 1em;
  margin-left: 0;
  padding-left: 0;
}

d-article li:last-child {
  margin-bottom: 0;
}

d-article pre {
  font-size: 14px;
  margin-bottom: 20px;
}

d-article hr {
  grid-column: screen;
  width: 100%;
  border: none;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  margin-top: 60px;
  margin-bottom: 60px;
}

d-article section {
  margin-top: 60px;
  margin-bottom: 60px;
}

d-article span.equation-mimic {
  font-family: georgia;
  font-size: 115%;
  font-style: italic;
}

d-article > d-code,
d-article section > d-code  {
  display: block;
}

d-article > d-math[block],
d-article section > d-math[block]  {
  display: block;
}

@media (max-width: 768px) {
  d-article > d-code,
  d-article section > d-code,
  d-article > d-math[block],
  d-article section > d-math[block] {
      overflow-x: scroll;
      -ms-overflow-style: none;  // IE 10+
      overflow: -moz-scrollbars-none;  // Firefox
  }

  d-article > d-code::-webkit-scrollbar,
  d-article section > d-code::-webkit-scrollbar,
  d-article > d-math[block]::-webkit-scrollbar,
  d-article section > d-math[block]::-webkit-scrollbar {
    display: none;  // Safari and Chrome
  }
}

d-article .citation {
  color: #668;
  cursor: pointer;
}

d-include {
  width: auto;
  display: block;
}

d-figure {
  contain: layout style;
}

/* KaTeX */

.katex, .katex-prerendered {
  contain: style;
  display: inline-block;
}

/* Tables */

d-article table {
  border-collapse: collapse;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.2);
}

d-article table th {
  border-bottom: 1px solid rgba(0, 0, 0, 0.2);
}

d-article table td {
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

d-article table tr:last-of-type td {
  border-bottom: none;
}

d-article table th,
d-article table td {
  font-size: 15px;
  padding: 2px 8px;
}

d-article table tbody :first-child td {
  padding-top: 2px;
}
/*
 * Copyright 2018 The Distill Template Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

span.katex-display {
  text-align: left;
  padding: 8px 0 8px 0;
  margin: 0.5em 0 0.5em 1em;
}

span.katex {
  -webkit-font-smoothing: antialiased;
  color: rgba(0, 0, 0, 0.8);
  font-size: 1.18em;
}
/*
 * Copyright 2018 The Distill Template Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

@media print {

  @page {
    size: 8in 11in;
    @bottom-right {
      content: counter(page) " of " counter(pages);
    }
  }

  html {
    /* no general margins -- CSS Grid takes care of those */
  }

  p, code {
    page-break-inside: avoid;
  }

  h2, h3 {
    page-break-after: avoid;
  }

  d-header {
    visibility: hidden;
  }

  d-footer {
    display: none!important;
  }

}
</style>
<style type="text/css">/* Chart.js */
@-webkit-keyframes chartjs-render-animation{from{opacity:0.99}to{opacity:1}}@keyframes chartjs-render-animation{from{opacity:0.99}to{opacity:1}}.chartjs-render-monitor{-webkit-animation:chartjs-render-animation 0.001s;animation:chartjs-render-animation 0.001s;}</style>
<style type="text/css">.medium-zoom-overlay{position:fixed;top:0;right:0;bottom:0;left:0;opacity:0;transition:opacity .3s;will-change:opacity}.medium-zoom--opened .medium-zoom-overlay{cursor:pointer;cursor:zoom-out;opacity:1}.medium-zoom-image{cursor:pointer;cursor:zoom-in;transition:transform .3s cubic-bezier(.2,0,.2,1)!important}.medium-zoom-image--hidden{visibility:hidden}.medium-zoom-image--opened{position:relative;cursor:pointer;cursor:zoom-out;will-change:transform}</style>
<style type="text/css">.CtxtMenu_InfoClose {  top:.2em; right:.2em;}
.CtxtMenu_InfoContent {  overflow:auto; text-align:left; font-size:80%;  padding:.4em .6em; border:1px inset; margin:1em 0px;  max-height:20em; max-width:30em; background-color:#EEEEEE;  white-space:normal;}
.CtxtMenu_Info.CtxtMenu_MousePost {outline:none;}
.CtxtMenu_Info {  position:fixed; left:50%; width:auto; text-align:center;  border:3px outset; padding:1em 2em; background-color:#DDDDDD;  color:black;  cursor:default; font-family:message-box; font-size:120%;  font-style:normal; text-indent:0; text-transform:none;  line-height:normal; letter-spacing:normal; word-spacing:normal;  word-wrap:normal; white-space:nowrap; float:none; z-index:201;  border-radius: 15px;                     /* Opera 10.5 and IE9 */  -webkit-border-radius:15px;               /* Safari and Chrome */  -moz-border-radius:15px;                  /* Firefox */  -khtml-border-radius:15px;                /* Konqueror */  box-shadow:0px 10px 20px #808080;         /* Opera 10.5 and IE9 */  -webkit-box-shadow:0px 10px 20px #808080; /* Safari 3 & Chrome */  -moz-box-shadow:0px 10px 20px #808080;    /* Forefox 3.5 */  -khtml-box-shadow:0px 10px 20px #808080;  /* Konqueror */  filter:progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color="gray", Positive="true"); /* IE */}
</style>
<style type="text/css">.CtxtMenu_MenuClose {  position:absolute;  cursor:pointer;  display:inline-block;  border:2px solid #AAA;  border-radius:18px;  -webkit-border-radius: 18px;             /* Safari and Chrome */  -moz-border-radius: 18px;                /* Firefox */  -khtml-border-radius: 18px;              /* Konqueror */  font-family: "Courier New", Courier;  font-size:24px;  color:#F0F0F0}
.CtxtMenu_MenuClose span {  display:block; background-color:#AAA; border:1.5px solid;  border-radius:18px;  -webkit-border-radius: 18px;             /* Safari and Chrome */  -moz-border-radius: 18px;                /* Firefox */  -khtml-border-radius: 18px;              /* Konqueror */  line-height:0;  padding:8px 0 6px     /* may need to be browser-specific */}
.CtxtMenu_MenuClose:hover {  color:white!important;  border:2px solid #CCC!important}
.CtxtMenu_MenuClose:hover span {  background-color:#CCC!important}
.CtxtMenu_MenuClose:hover:focus {  outline:none}
</style>
<style type="text/css">.CtxtMenu_Menu {  position:absolute;  background-color:white;  color:black;  width:auto; padding:5px 0px;  border:1px solid #CCCCCC; margin:0; cursor:default;  font: menu; text-align:left; text-indent:0; text-transform:none;  line-height:normal; letter-spacing:normal; word-spacing:normal;  word-wrap:normal; white-space:nowrap; float:none; z-index:201;  border-radius: 5px;                     /* Opera 10.5 and IE9 */  -webkit-border-radius: 5px;             /* Safari and Chrome */  -moz-border-radius: 5px;                /* Firefox */  -khtml-border-radius: 5px;              /* Konqueror */  box-shadow:0px 10px 20px #808080;         /* Opera 10.5 and IE9 */  -webkit-box-shadow:0px 10px 20px #808080; /* Safari 3 & Chrome */  -moz-box-shadow:0px 10px 20px #808080;    /* Forefox 3.5 */  -khtml-box-shadow:0px 10px 20px #808080;  /* Konqueror */}
.CtxtMenu_MenuItem {  padding: 1px 2em;  background:transparent;}
.CtxtMenu_MenuArrow {  position:absolute; right:.5em; padding-top:.25em; color:#666666;  font-family: null; font-size: .75em}
.CtxtMenu_MenuActive .CtxtMenu_MenuArrow {color:white}
.CtxtMenu_MenuArrow.CtxtMenu_RTL {left:.5em; right:auto}
.CtxtMenu_MenuCheck {  position:absolute; left:.7em;  font-family: null}
.CtxtMenu_MenuCheck.CtxtMenu_RTL { right:.7em; left:auto }
.CtxtMenu_MenuRadioCheck {  position:absolute; left: .7em;}
.CtxtMenu_MenuRadioCheck.CtxtMenu_RTL {  right: .7em; left:auto}
.CtxtMenu_MenuInputBox {  padding-left: 1em; right:.5em; color:#666666;  font-family: null;}
.CtxtMenu_MenuInputBox.CtxtMenu_RTL {  left: .1em;}
.CtxtMenu_MenuComboBox {  left:.1em; padding-bottom:.5em;}
.CtxtMenu_MenuSlider {  left: .1em;}
.CtxtMenu_SliderValue {  position:absolute; right:.1em; padding-top:.25em; color:#333333;  font-size: .75em}
.CtxtMenu_SliderBar {  outline: none; background: #d3d3d3}
.CtxtMenu_MenuLabel {  padding: 1px 2em 3px 1.33em;  font-style:italic}
.CtxtMenu_MenuRule {  border-top: 1px solid #DDDDDD;  margin: 4px 3px;}
.CtxtMenu_MenuDisabled {  color:GrayText}
.CtxtMenu_MenuActive {  background-color: #606872;  color: white;}
.CtxtMenu_MenuDisabled:focus {  background-color: #E8E8E8}
.CtxtMenu_MenuLabel:focus {  background-color: #E8E8E8}
.CtxtMenu_ContextMenu:focus {  outline:none}
.CtxtMenu_ContextMenu .CtxtMenu_MenuItem:focus {  outline:none}
.CtxtMenu_SelectionMenu {  position:relative; float:left;  border-bottom: none; -webkit-box-shadow:none; -webkit-border-radius:0px; }
.CtxtMenu_SelectionItem {  padding-right: 1em;}
.CtxtMenu_Selection {  right: 40%; width:50%; }
.CtxtMenu_SelectionBox {  padding: 0em; max-height:20em; max-width: none;  background-color:#FFFFFF;}
.CtxtMenu_SelectionDivider {  clear: both; border-top: 2px solid #000000;}
.CtxtMenu_Menu .CtxtMenu_MenuClose {  top:-10px; left:-10px}
</style>
<style id="MJX-CHTML-styles">
mjx-container[jax="CHTML"] {
  line-height: 0;
}

mjx-container [space="1"] {
  margin-left: .111em;
}

mjx-container [space="2"] {
  margin-left: .167em;
}

mjx-container [space="3"] {
  margin-left: .222em;
}

mjx-container [space="4"] {
  margin-left: .278em;
}

mjx-container [space="5"] {
  margin-left: .333em;
}

mjx-container [rspace="1"] {
  margin-right: .111em;
}

mjx-container [rspace="2"] {
  margin-right: .167em;
}

mjx-container [rspace="3"] {
  margin-right: .222em;
}

mjx-container [rspace="4"] {
  margin-right: .278em;
}

mjx-container [rspace="5"] {
  margin-right: .333em;
}

mjx-container [size="s"] {
  font-size: 70.7%;
}

mjx-container [size="ss"] {
  font-size: 50%;
}

mjx-container [size="Tn"] {
  font-size: 60%;
}

mjx-container [size="sm"] {
  font-size: 85%;
}

mjx-container [size="lg"] {
  font-size: 120%;
}

mjx-container [size="Lg"] {
  font-size: 144%;
}

mjx-container [size="LG"] {
  font-size: 173%;
}

mjx-container [size="hg"] {
  font-size: 207%;
}

mjx-container [size="HG"] {
  font-size: 249%;
}

mjx-container [width="full"] {
  width: 100%;
}

mjx-box {
  display: inline-block;
}

mjx-block {
  display: block;
}

mjx-itable {
  display: inline-table;
}

mjx-row {
  display: table-row;
}

mjx-row > * {
  display: table-cell;
}

mjx-mtext {
  display: inline-block;
  text-align: left;
}

mjx-mstyle {
  display: inline-block;
}

mjx-merror {
  display: inline-block;
  color: red;
  background-color: yellow;
}

mjx-mphantom {
  visibility: hidden;
}

_::-webkit-full-page-media, _:future, :root mjx-container {
  will-change: opacity;
}

mjx-assistive-mml {
  position: absolute !important;
  top: 0px;
  left: 0px;
  clip: rect(1px, 1px, 1px, 1px);
  padding: 1px 0px 0px 0px !important;
  border: 0px !important;
  display: block !important;
  width: auto !important;
  overflow: hidden !important;
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

mjx-assistive-mml[display="block"] {
  width: 100% !important;
}

mjx-math {
  display: inline-block;
  text-align: left;
  line-height: 0;
  text-indent: 0;
  font-style: normal;
  font-weight: normal;
  font-size: 100%;
  font-size-adjust: none;
  letter-spacing: normal;
  border-collapse: collapse;
  word-wrap: normal;
  word-spacing: normal;
  white-space: nowrap;
  direction: ltr;
  padding: 1px 0;
}

mjx-container[jax="CHTML"][display="true"] {
  display: block;
  text-align: center;
  margin: 1em 0;
}

mjx-container[jax="CHTML"][display="true"][width="full"] {
  display: flex;
}

mjx-container[jax="CHTML"][display="true"] mjx-math {
  padding: 0;
}

mjx-container[jax="CHTML"][justify="left"] {
  text-align: left;
}

mjx-container[jax="CHTML"][justify="right"] {
  text-align: right;
}

mjx-mi {
  display: inline-block;
  text-align: left;
}

mjx-c {
  display: inline-block;
}

mjx-utext {
  display: inline-block;
  padding: .75em 0 .2em 0;
}

mjx-mo {
  display: inline-block;
  text-align: left;
}

mjx-stretchy-h {
  display: inline-table;
  width: 100%;
}

mjx-stretchy-h > * {
  display: table-cell;
  width: 0;
}

mjx-stretchy-h > * > mjx-c {
  display: inline-block;
  transform: scalex(1.0000001);
}

mjx-stretchy-h > * > mjx-c::before {
  display: inline-block;
  width: initial;
}

mjx-stretchy-h > mjx-ext {
  /* IE */ overflow: hidden;
  /* others */ overflow: clip visible;
  width: 100%;
}

mjx-stretchy-h > mjx-ext > mjx-c::before {
  transform: scalex(500);
}

mjx-stretchy-h > mjx-ext > mjx-c {
  width: 0;
}

mjx-stretchy-h > mjx-beg > mjx-c {
  margin-right: -.1em;
}

mjx-stretchy-h > mjx-end > mjx-c {
  margin-left: -.1em;
}

mjx-stretchy-v {
  display: inline-block;
}

mjx-stretchy-v > * {
  display: block;
}

mjx-stretchy-v > mjx-beg {
  height: 0;
}

mjx-stretchy-v > mjx-end > mjx-c {
  display: block;
}

mjx-stretchy-v > * > mjx-c {
  transform: scaley(1.0000001);
  transform-origin: left center;
  overflow: hidden;
}

mjx-stretchy-v > mjx-ext {
  display: block;
  height: 100%;
  box-sizing: border-box;
  border: 0px solid transparent;
  /* IE */ overflow: hidden;
  /* others */ overflow: visible clip;
}

mjx-stretchy-v > mjx-ext > mjx-c::before {
  width: initial;
  box-sizing: border-box;
}

mjx-stretchy-v > mjx-ext > mjx-c {
  transform: scaleY(500) translateY(.075em);
  overflow: visible;
}

mjx-mark {
  display: inline-block;
  height: 0px;
}

mjx-TeXAtom {
  display: inline-block;
  text-align: left;
}

mjx-mn {
  display: inline-block;
  text-align: left;
}

mjx-mfrac {
  display: inline-block;
  text-align: left;
}

mjx-frac {
  display: inline-block;
  vertical-align: 0.17em;
  padding: 0 .22em;
}

mjx-frac[type="d"] {
  vertical-align: .04em;
}

mjx-frac[delims] {
  padding: 0 .1em;
}

mjx-frac[atop] {
  padding: 0 .12em;
}

mjx-frac[atop][delims] {
  padding: 0;
}

mjx-dtable {
  display: inline-table;
  width: 100%;
}

mjx-dtable > * {
  font-size: 2000%;
}

mjx-dbox {
  display: block;
  font-size: 5%;
}

mjx-num {
  display: block;
  text-align: center;
}

mjx-den {
  display: block;
  text-align: center;
}

mjx-mfrac[bevelled] > mjx-num {
  display: inline-block;
}

mjx-mfrac[bevelled] > mjx-den {
  display: inline-block;
}

mjx-den[align="right"], mjx-num[align="right"] {
  text-align: right;
}

mjx-den[align="left"], mjx-num[align="left"] {
  text-align: left;
}

mjx-nstrut {
  display: inline-block;
  height: .054em;
  width: 0;
  vertical-align: -.054em;
}

mjx-nstrut[type="d"] {
  height: .217em;
  vertical-align: -.217em;
}

mjx-dstrut {
  display: inline-block;
  height: .505em;
  width: 0;
}

mjx-dstrut[type="d"] {
  height: .726em;
}

mjx-line {
  display: block;
  box-sizing: border-box;
  min-height: 1px;
  height: .06em;
  border-top: .06em solid;
  margin: .06em -.1em;
  overflow: hidden;
}

mjx-line[type="d"] {
  margin: .18em -.1em;
}

mjx-mrow {
  display: inline-block;
  text-align: left;
}

mjx-msup {
  display: inline-block;
  text-align: left;
}

mjx-msub {
  display: inline-block;
  text-align: left;
}

mjx-msubsup {
  display: inline-block;
  text-align: left;
}

mjx-script {
  display: inline-block;
  padding-right: .05em;
  padding-left: .033em;
}

mjx-script > mjx-spacer {
  display: block;
}

mjx-c::before {
  display: block;
  width: 0;
}

.MJX-TEX {
  font-family: MJXZERO, MJXTEX;
}

.TEX-B {
  font-family: MJXZERO, MJXTEX-B;
}

.TEX-I {
  font-family: MJXZERO, MJXTEX-I;
}

.TEX-MI {
  font-family: MJXZERO, MJXTEX-MI;
}

.TEX-BI {
  font-family: MJXZERO, MJXTEX-BI;
}

.TEX-S1 {
  font-family: MJXZERO, MJXTEX-S1;
}

.TEX-S2 {
  font-family: MJXZERO, MJXTEX-S2;
}

.TEX-S3 {
  font-family: MJXZERO, MJXTEX-S3;
}

.TEX-S4 {
  font-family: MJXZERO, MJXTEX-S4;
}

.TEX-A {
  font-family: MJXZERO, MJXTEX-A;
}

.TEX-C {
  font-family: MJXZERO, MJXTEX-C;
}

.TEX-CB {
  font-family: MJXZERO, MJXTEX-CB;
}

.TEX-FR {
  font-family: MJXZERO, MJXTEX-FR;
}

.TEX-FRB {
  font-family: MJXZERO, MJXTEX-FRB;
}

.TEX-SS {
  font-family: MJXZERO, MJXTEX-SS;
}

.TEX-SSB {
  font-family: MJXZERO, MJXTEX-SSB;
}

.TEX-SSI {
  font-family: MJXZERO, MJXTEX-SSI;
}

.TEX-SC {
  font-family: MJXZERO, MJXTEX-SC;
}

.TEX-T {
  font-family: MJXZERO, MJXTEX-T;
}

.TEX-V {
  font-family: MJXZERO, MJXTEX-V;
}

.TEX-VB {
  font-family: MJXZERO, MJXTEX-VB;
}

mjx-stretchy-v mjx-c, mjx-stretchy-h mjx-c {
  font-family: MJXZERO, MJXTEX-S1, MJXTEX-S4, MJXTEX, MJXTEX-A ! important;
}

@font-face /* 0 */ {
  font-family: MJXZERO;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Zero.woff") format("woff");
}

@font-face /* 1 */ {
  font-family: MJXTEX;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff") format("woff");
}

@font-face /* 2 */ {
  font-family: MJXTEX-B;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Main-Bold.woff") format("woff");
}

@font-face /* 3 */ {
  font-family: MJXTEX-I;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Math-Italic.woff") format("woff");
}

@font-face /* 4 */ {
  font-family: MJXTEX-MI;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Main-Italic.woff") format("woff");
}

@font-face /* 5 */ {
  font-family: MJXTEX-BI;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Math-BoldItalic.woff") format("woff");
}

@font-face /* 6 */ {
  font-family: MJXTEX-S1;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Size1-Regular.woff") format("woff");
}

@font-face /* 7 */ {
  font-family: MJXTEX-S2;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Size2-Regular.woff") format("woff");
}

@font-face /* 8 */ {
  font-family: MJXTEX-S3;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Size3-Regular.woff") format("woff");
}

@font-face /* 9 */ {
  font-family: MJXTEX-S4;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Size4-Regular.woff") format("woff");
}

@font-face /* 10 */ {
  font-family: MJXTEX-A;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_AMS-Regular.woff") format("woff");
}

@font-face /* 11 */ {
  font-family: MJXTEX-C;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Calligraphic-Regular.woff") format("woff");
}

@font-face /* 12 */ {
  font-family: MJXTEX-CB;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Calligraphic-Bold.woff") format("woff");
}

@font-face /* 13 */ {
  font-family: MJXTEX-FR;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Fraktur-Regular.woff") format("woff");
}

@font-face /* 14 */ {
  font-family: MJXTEX-FRB;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Fraktur-Bold.woff") format("woff");
}

@font-face /* 15 */ {
  font-family: MJXTEX-SS;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_SansSerif-Regular.woff") format("woff");
}

@font-face /* 16 */ {
  font-family: MJXTEX-SSB;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_SansSerif-Bold.woff") format("woff");
}

@font-face /* 17 */ {
  font-family: MJXTEX-SSI;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_SansSerif-Italic.woff") format("woff");
}

@font-face /* 18 */ {
  font-family: MJXTEX-SC;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Script-Regular.woff") format("woff");
}

@font-face /* 19 */ {
  font-family: MJXTEX-T;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Typewriter-Regular.woff") format("woff");
}

@font-face /* 20 */ {
  font-family: MJXTEX-V;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Vector-Regular.woff") format("woff");
}

@font-face /* 21 */ {
  font-family: MJXTEX-VB;
  src: url("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2/MathJax_Vector-Bold.woff") format("woff");
}

mjx-c.mjx-c1D43E.TEX-I::before {
  padding: 0.683em 0.889em 0 0;
  content: "K";
}

mjx-c.mjx-c1D438.TEX-I::before {
  padding: 0.68em 0.764em 0 0;
  content: "E";
}

mjx-c.mjx-c1D43A.TEX-I::before {
  padding: 0.705em 0.786em 0.022em 0;
  content: "G";
}

mjx-c.mjx-c3D::before {
  padding: 0.583em 0.778em 0.082em 0;
  content: "=";
}

mjx-c.mjx-c1D451.TEX-I::before {
  padding: 0.694em 0.52em 0.01em 0;
  content: "d";
}

mjx-c.mjx-c2F::before {
  padding: 0.75em 0.5em 0.25em 0;
  content: "/";
}

mjx-c.mjx-c1D45B.TEX-I::before {
  padding: 0.442em 0.6em 0.011em 0;
  content: "n";
}

mjx-c.mjx-c1D70C.TEX-I::before {
  padding: 0.442em 0.517em 0.216em 0;
  content: "\3C1";
}

mjx-c.mjx-c30::before {
  padding: 0.666em 0.5em 0.022em 0;
  content: "0";
}

mjx-c.mjx-c2E::before {
  padding: 0.12em 0.278em 0 0;
  content: ".";
}

mjx-c.mjx-c33::before {
  padding: 0.665em 0.5em 0.022em 0;
  content: "3";
}

mjx-c.mjx-c38::before {
  padding: 0.666em 0.5em 0.022em 0;
  content: "8";
}

mjx-c.mjx-c32::before {
  padding: 0.666em 0.5em 0 0;
  content: "2";
}

mjx-c.mjx-c35::before {
  padding: 0.666em 0.5em 0.022em 0;
  content: "5";
}

mjx-c.mjx-c34::before {
  padding: 0.677em 0.5em 0 0;
  content: "4";
}

mjx-c.mjx-c28::before {
  padding: 0.75em 0.389em 0.25em 0;
  content: "(";
}

mjx-c.mjx-c36::before {
  padding: 0.666em 0.5em 0.022em 0;
  content: "6";
}

mjx-c.mjx-c2B::before {
  padding: 0.583em 0.778em 0.082em 0;
  content: "+";
}

mjx-c.mjx-c31::before {
  padding: 0.666em 0.5em 0 0;
  content: "1";
}

mjx-c.mjx-c29::before {
  padding: 0.75em 0.389em 0.25em 0;
  content: ")";
}

mjx-c.mjx-c1D447.TEX-I::before {
  padding: 0.677em 0.704em 0 0;
  content: "T";
}

mjx-c.mjx-c1D442.TEX-I::before {
  padding: 0.704em 0.763em 0.022em 0;
  content: "O";
}

mjx-c.mjx-c1D44C.TEX-I::before {
  padding: 0.683em 0.763em 0 0;
  content: "Y";
}

mjx-c.mjx-c41::before {
  padding: 0.716em 0.75em 0 0;
  content: "A";
}

mjx-c.mjx-c72::before {
  padding: 0.442em 0.392em 0 0;
  content: "r";
}

mjx-c.mjx-c69::before {
  padding: 0.669em 0.278em 0 0;
  content: "i";
}

mjx-c.mjx-c74::before {
  padding: 0.615em 0.389em 0.01em 0;
  content: "t";
}

mjx-c.mjx-c68::before {
  padding: 0.694em 0.556em 0 0;
  content: "h";
}

mjx-c.mjx-c6D::before {
  padding: 0.442em 0.833em 0 0;
  content: "m";
}

mjx-c.mjx-c65::before {
  padding: 0.448em 0.444em 0.011em 0;
  content: "e";
}

mjx-c.mjx-c63::before {
  padding: 0.448em 0.444em 0.011em 0;
  content: "c";
}

mjx-c.mjx-c20::before {
  padding: 0 0.25em 0 0;
  content: " ";
}

mjx-c.mjx-c49::before {
  padding: 0.683em 0.361em 0 0;
  content: "I";
}

mjx-c.mjx-c6E::before {
  padding: 0.442em 0.556em 0 0;
  content: "n";
}

mjx-c.mjx-c73::before {
  padding: 0.448em 0.394em 0.011em 0;
  content: "s";
}

mjx-c.mjx-c79::before {
  padding: 0.431em 0.528em 0.204em 0;
  content: "y";
}

mjx-c.mjx-c28.TEX-S3::before {
  padding: 1.45em 0.736em 0.949em 0;
  content: "(";
}

mjx-c.mjx-c2C::before {
  padding: 0.121em 0.278em 0.194em 0;
  content: ",";
}

mjx-c.mjx-c29.TEX-S3::before {
  padding: 1.45em 0.736em 0.949em 0;
  content: ")";
}

mjx-c.mjx-c1D436.TEX-I::before {
  padding: 0.705em 0.76em 0.022em 0;
  content: "C";
}

mjx-c.mjx-c1D434.TEX-I::before {
  padding: 0.716em 0.75em 0 0;
  content: "A";
}

mjx-c.mjx-c1D435.TEX-I::before {
  padding: 0.683em 0.759em 0 0;
  content: "B";
}

mjx-c.mjx-c2208::before {
  padding: 0.54em 0.667em 0.04em 0;
  content: "\2208";
}

mjx-c.mjx-c211D.TEX-A::before {
  padding: 0.683em 0.722em 0 0;
  content: "R";
}

mjx-c.mjx-c1D440.TEX-I::before {
  padding: 0.683em 1.051em 0 0;
  content: "M";
}

mjx-c.mjx-cD7::before {
  padding: 0.491em 0.778em 0 0;
  content: "\D7";
}

mjx-c.mjx-c1D441.TEX-I::before {
  padding: 0.683em 0.888em 0 0;
  content: "N";
}

mjx-c.mjx-c1D44B.TEX-I::before {
  padding: 0.683em 0.852em 0 0;
  content: "X";
}

mjx-c.mjx-c1D43B.TEX-I::before {
  padding: 0.683em 0.888em 0 0;
  content: "H";
}

mjx-c.mjx-c1D446.TEX-I::before {
  padding: 0.705em 0.645em 0.022em 0;
  content: "S";
}

mjx-c.mjx-c2032::before {
  padding: 0.56em 0.275em 0 0;
  content: "\2032";
}

mjx-c.mjx-c1D452.TEX-I::before {
  padding: 0.442em 0.466em 0.011em 0;
  content: "e";
}

mjx-c.mjx-c1D44A.TEX-I::before {
  padding: 0.683em 1.048em 0.022em 0;
  content: "W";
}

mjx-c.mjx-c22A4::before {
  padding: 0.668em 0.778em 0 0;
  content: "\22A4";
}

mjx-c.mjx-c1D461.TEX-I::before {
  padding: 0.626em 0.361em 0.011em 0;
  content: "t";
}

mjx-c.mjx-c27E8::before {
  padding: 0.75em 0.389em 0.25em 0;
  content: "\27E8";
}

mjx-c.mjx-cA0::before {
  padding: 0 0.25em 0 0;
  content: "\A0";
}

mjx-c.mjx-c27E9::before {
  padding: 0.75em 0.389em 0.25em 0;
  content: "\27E9";
}

mjx-c.mjx-c6C::before {
  padding: 0.694em 0.278em 0 0;
  content: "l";
}

mjx-c.mjx-c37::before {
  padding: 0.676em 0.5em 0.022em 0;
  content: "7";
}

mjx-c.mjx-c39::before {
  padding: 0.666em 0.5em 0.022em 0;
  content: "9";
}
</style>
<style>
          .mjx-container {
            color: inherit;
          }
        </style>
<style id="zh-tr-style">
.zh-tr {
  background: #f0f7ff; border-left: 3px solid #4a90e2;
  padding: 8px 14px !important; margin: 6px 0 14px 0 !important;
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 0.96em; line-height: 1.7; color: #1a3a5c; border-radius: 3px;
}
.zh-tr code { background: #d8e6f5; }
h1.zh-h, h2.zh-h, h3.zh-h, h4.zh-h, h5.zh-h {
  background: #f0f7ff; color: #1a3a5c !important;
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  font-weight: 600; font-size: 0.85em !important;
  margin-top: -8px !important; padding: 4px 10px !important;
  border-left: 3px solid #4a90e2; border-radius: 0 3px 3px 0; border-bottom: none !important;
}
.zh-banner {
  background: linear-gradient(90deg,#f0f7ff,#fff);
  border-left: 4px solid #4a90e2;
  padding: 10px 16px; margin: 16px 0 24px; font-size: 14px; color: #1a3a5c;
}
.zh-banner b { color: #003366; }

.deep-dive {
  background: #eef7ee; border-left: 4px solid #5fa55f;
  margin: 16px 0 22px 0; padding: 14px 18px; border-radius: 4px;
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 0.95em; line-height: 1.75; color: #1a3d1a;
}
.deep-dive .dd-label {
  display: inline-block; background: #5fa55f; color: white;
  font-size: 12px; font-weight: 700; padding: 2px 10px;
  border-radius: 3px; letter-spacing: 0.5px; margin-bottom: 8px;
}
.deep-dive strong { display: block; font-size: 1.05em; color: #0f3d0f; margin-bottom: 8px; }
.deep-dive code { background: #d7e8d7; color: #0f3d0f; padding: 1px 5px; border-radius: 3px; font-size: 0.92em; }
.deep-dive p { margin: 8px 0; }
.deep-dive ol, .deep-dive ul { margin: 6px 0; padding-left: 24px; }
.deep-dive li { margin: 4px 0; }
.deep-dive table { border-collapse: collapse; margin: 10px 0; }
.deep-dive pre { font-family: "SF Mono", Menlo, Consolas, monospace; }

/* Prologue (background + notation) — full-bleed breakout so it can show wide tables/SVGs */
.prologue {
  background: #fff8e7;
  border: 1px solid #e0b300;
  border-left: 5px solid #e0b300;
  margin: 20px 0 30px;
  padding: 20px 28px 24px;
  border-radius: 4px;
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 0.97em; line-height: 1.75; color: #4a3500;

  /* Break out of the narrow Distill column: center the block to the viewport,
     capped at 1500px on wide monitors, shrinking on narrow screens. */
  position: relative;
  width: min(1500px, 94vw);
  max-width: none;
  left: 50%;
  transform: translateX(-50%);
  box-sizing: border-box;
}
@media (max-width: 900px) {
  .prologue {
    /* On narrow screens fall back to container-width to avoid horizontal scroll */
    width: auto;
    left: 0;
    transform: none;
    padding: 14px 16px;
  }
}
.prologue-title {
  margin: 0 0 10px !important; color: #7a4e00 !important;
  font-size: 20px !important; border-bottom: 2px solid #e0b300; padding-bottom: 6px !important;
}
.prologue-intro { margin: 8px 0 12px; color: #5a3f00; font-size: 14px; }
.prologue-h3 {
  color: #7a4e00 !important; margin: 18px 0 8px !important;
  font-size: 15.5px !important; border-bottom: 1px dashed #e0b300; padding-bottom: 3px;
}
.prologue-h4 { color: #7a4e00; margin: 12px 0 6px; font-size: 14px; }
.prologue .prologue-tbl { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13.5px; }
.prologue .prologue-tbl th, .prologue .prologue-tbl td {
  border: 1px solid #d9b860; padding: 6px 10px; text-align: left; vertical-align: top;
}
.prologue .prologue-tbl th { background: #fff1c4; color: #5a3f00; font-weight: 600; }
.prologue .prologue-tbl td { background: #fffcf1; }
.prologue code {
  background: #fff1c4; color: #5a3f00; padding: 1px 5px;
  border-radius: 3px; font-size: 0.9em;
}
.prologue ol, .prologue ul { padding-left: 26px; margin: 6px 0; }
.prologue li { margin: 4px 0; }
.prologue-note {
  background: #fff2cc; border-left: 4px solid #d6b656;
  padding: 8px 12px; margin: 10px 0; font-size: 0.95em;
}
.prologue-foot {
  background: #fff5d8; border-left: 4px solid #e0b300;
  padding: 8px 12px; margin: 16px 0 0; font-size: 0.95em;
}
.prologue-toc {
  background: #fffcf1;
  border: 1px solid #d9b860;
  border-radius: 4px;
  padding: 12px 20px 14px;
  margin: 10px 0 20px;
  font-size: 13.5px;
  line-height: 1.7;
}
.prologue-toc ol {
  margin: 8px 0 6px;
  padding-left: 26px;
  color: #4a3500;
}
.prologue-toc ol li { margin: 3px 0; }
.prologue-toc a {
  color: #7a4e00;
  text-decoration: none;
  font-weight: 600;
}
.prologue-toc a:hover { text-decoration: underline; color: #b46504; }
.prologue-toc .toc-sub {
  color: #8a6f2f;
  font-size: 0.9em;
  font-weight: normal;
  margin-left: 6px;
}
.prologue-toc .toc-new {
  display: inline-block;
  background: #d6336c;
  color: #fff;
  font-size: 10px;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 3px;
  margin-left: 6px;
  vertical-align: middle;
}
.prologue-toc .toc-tip {
  margin: 10px 0 0;
  padding: 8px 12px;
  background: #fff5d8;
  border-left: 3px solid #e0b300;
  color: #5a3f00;
  font-size: 12.5px;
  border-radius: 3px;
}
.svg-wrapper {
  margin: 10px 0 16px;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}
.svg-wrapper svg {
  min-width: 720px;
  display: block;
  width: 100%;
  height: auto;
}
.prologue .prologue-tbl { font-size: 13.8px; }
.prologue .prologue-tbl td, .prologue .prologue-tbl th { padding: 7px 12px; }

/* Dedicated stepwise comparison table — even denser, wider */
.stepwise-tbl {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0 4px;
  font-size: 12.5px;
  line-height: 1.55;
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  table-layout: fixed;
}
.stepwise-tbl th {
  background: #fff1c4;
  color: #5a3f00;
  font-weight: 700;
  border: 1px solid #d9b860;
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}
.stepwise-tbl td {
  background: #fffcf1;
  border: 1px solid #d9b860;
  padding: 7px 10px;
  vertical-align: top;
  word-break: break-word;
  overflow-wrap: anywhere;
}
.stepwise-tbl td:nth-child(1) {
  text-align: center;
  font-weight: 700;
  background: #fff1c4;
  color: #7a4e00;
}
.stepwise-tbl td:nth-child(2) { font-weight: 600; }
.stepwise-tbl td:nth-child(5) { text-align: center; font-size: 16px; }
.stepwise-tbl code {
  background: #fff1c4;
  color: #5a3f00;
  padding: 0 4px;
  border-radius: 2px;
  font-size: 11.5px;
}

/* Formula boxes under each SVG */
.formula-box {
  margin: 6px 0 16px;
  padding: 12px 18px;
  border-radius: 4px;
  font-size: 13.5px;
  line-height: 1.7;
}
.formula-box.std-box {
  background: #fff5f0;
  border: 1px solid #b85450;
  border-left: 4px solid #b85450;
  color: #4a1515;
}
.formula-box.sm-box {
  background: #f4faf4;
  border: 1px solid #5fa55f;
  border-left: 4px solid #5fa55f;
  color: #1a3d1a;
}
.formula-box .formula-label {
  display: inline-block;
  font-weight: 700;
  font-size: 12.5px;
  padding: 2px 10px;
  border-radius: 3px;
  margin-bottom: 6px;
  letter-spacing: 0.3px;
}
.formula-box.std-box .formula-label { background: #b85450; color: #fff; }
.formula-box.sm-box  .formula-label { background: #5fa55f; color: #fff; }
.formula-box code {
  background: rgba(0,0,0,0.08);
  color: inherit;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 0.9em;
}
.formula-box p { margin: 6px 0; }

/* Equivalence proof box */
.eq-box {
  margin: 18px 0 20px;
  padding: 16px 22px 18px;
  border-radius: 5px;
  background: linear-gradient(135deg, #fff5f0 0%, #fffcf1 50%, #f4faf4 100%);
  border: 1px solid #c9a26b;
  border-left: 5px solid #c9a26b;
  font-size: 13.5px;
  line-height: 1.75;
  color: #3a2f15;
}
.eq-box .eq-label {
  display: inline-block;
  background: #8a5a00;
  color: #fff;
  font-weight: 700;
  font-size: 13px;
  padding: 3px 12px;
  border-radius: 3px;
  margin-bottom: 10px;
  letter-spacing: 0.3px;
}
.eq-box .eq-step {
  background: #fffcf1;
  border: 1px solid #d9b860;
  border-radius: 4px;
  padding: 10px 14px;
  margin: 10px 0;
}
.eq-box .eq-step-title {
  font-weight: 700;
  color: #7a4e00;
  font-size: 14px;
  margin-bottom: 4px;
  padding-bottom: 3px;
  border-bottom: 1px dashed #d9b860;
}
.eq-box .eq-concl {
  margin: 6px 0 0;
  padding: 6px 10px;
  background: #fff5d8;
  border-left: 3px solid #e0b300;
  color: #5a3f00;
  font-size: 12.5px;
  border-radius: 2px;
}
.eq-box code {
  background: #fff1c4;
  color: #5a3f00;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 0.9em;
}

/* Formula list (bulleted equations, using inline $...$ math only) */
.fml-list {
  margin: 6px 0 8px;
  padding-left: 24px;
  font-size: 14px;
  line-height: 2.0;
}
.fml-list li { margin: 2px 0; }

/* Formula table: 2-column (equation | note) or 3-column (source | equation | note) */
.fml-tbl {
  border-collapse: collapse;
  width: 100%;
  margin: 6px 0 10px;
  font-size: 13.5px;
  line-height: 1.85;
}
.fml-tbl td {
  padding: 5px 10px;
  vertical-align: middle;
  border: 1px solid rgba(0,0,0,0.1);
}
.fml-tbl.std td { background: #fff9f6; }
.fml-tbl.sm td  { background: #f9fdf9; }
.fml-tbl.derive td { background: #fffcf1; }

.fml-tbl .fml-eq {
  font-size: 14.5px;
  padding: 6px 12px;
}
.fml-tbl .fml-note {
  font-size: 12px;
  color: #666;
  width: 32%;
  text-align: left;
}
.fml-tbl .fml-src {
  width: 18%;
  font-size: 12.5px;
  color: #333;
  font-weight: 600;
  text-align: right;
  padding-right: 12px;
}
.fml-tbl .fml-src.std { color: #b85450; }
.fml-tbl .fml-src.sm  { color: #1f5d1f; }
.fml-tbl code { font-size: 0.9em; }
</style>
 <d-front-matter> <script async="" type="text/json">
      {
            "title": "SonicMoE: A Hardware-Efficient and Software-Extensible Blueprint for Fine-Grained MoEs",
            "description": "",
            "published": "April 22, 2026",
            "authors": [
              
              {
                "author": "Wentao Guo",
                "authorURL": "",
                "affiliations": [
                  {
                    "name": "Princeton University",
                    "url": ""
                  }
                ]
              },
              
              {
                "author": "Mayank Mishra",
                "authorURL": "",
                "affiliations": [
                  {
                    "name": "UC Berkeley",
                    "url": ""
                  }
                ]
              },
              
              {
                "author": "Xinle Cheng",
                "authorURL": "",
                "affiliations": [
                  {
                    "name": "Princeton University",
                    "url": ""
                  }
                ]
              },
              
              {
                "author": "Ion Stoica",
                "authorURL": "",
                "affiliations": [
                  {
                    "name": "UC Berkeley",
                    "url": ""
                  }
                ]
              },
              
              {
                "author": "Tri Dao",
                "authorURL": "",
                "affiliations": [
                  {
                    "name": "Princeton University",
                    "url": ""
                  }
                ]
              }
              
            ],
            "katex": {
              "delimiters": [
                {
                  "left": "$",
                  "right": "$",
                  "display": false
                },
                {
                  "left": "$$",
                  "right": "$$",
                  "display": true
                }
              ]
            }
          }
    </script> </d-front-matter> <header> <nav id="navbar" class="navbar navbar-light navbar-expand-sm fixed-top" role="navigation"> <div class="container"> <a class="navbar-brand title font-weight-lighter" href="https://dao-lab.ai/"> Dao AI Lab </a> <button class="navbar-toggler collapsed ml-auto" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation"> <span class="sr-only">Toggle navigation</span> <span class="icon-bar top-bar"></span> <span class="icon-bar middle-bar"></span> <span class="icon-bar bottom-bar"></span> </button> <div class="collapse navbar-collapse text-right" id="navbarNav"> <ul class="navbar-nav ml-auto flex-nowrap"> <li class="nav-item "> <a class="nav-link" href="https://dao-lab.ai/">Research Group </a> </li> <li class="nav-item "> <a class="nav-link" href="https://dao-lab.ai/publications/">publications </a> </li> <li class="nav-item active"> <a class="nav-link" href="https://dao-lab.ai/blog/">blog </a> </li> <li class="nav-item "> <a class="nav-link" href="https://dao-lab.ai/repositories/">Repositories </a> </li> <li class="toggle-container"> <button id="light-toggle" title="Change theme"> <i class="fa-half-sun-moon" id="light-toggle-system"></i> <i class="fa-solid fa-moon" id="light-toggle-dark"></i> <i class="fa-solid fa-sun" id="light-toggle-light"></i> </button> </li> </ul> </div> </div> </nav> <progress id="progress" value="33945" max="34496" style="top: 57px;"> <div class="progress-container"> <span class="progress-bar"></span> </div> </progress> </header> <div class="post distill"> <d-title> <h1>SonicMoE: A Hardware-Efficient and Software-Extensible Blueprint for Fine-Grained MoEs</h1> <p></p> </d-title> <d-byline>
  <div class="byline grid">
    <div class="authors-affiliations grid">
      <h3>Authors</h3>
      <h3>Affiliations</h3>
      
        <p class="author">
          
            <span class="name">Wentao Guo</span>
        </p>
        <p class="affiliation">
        <span class="affiliation">Princeton University</span>
        </p>
      
        <p class="author">
          
            <span class="name">Mayank Mishra</span>
        </p>
        <p class="affiliation">
        <span class="affiliation">UC Berkeley</span>
        </p>
      
        <p class="author">
          
            <span class="name">Xinle Cheng</span>
        </p>
        <p class="affiliation">
        <span class="affiliation">Princeton University</span>
        </p>
      
        <p class="author">
          
            <span class="name">Ion Stoica</span>
        </p>
        <p class="affiliation">
        <span class="affiliation">UC Berkeley</span>
        </p>
      
        <p class="author">
          
            <span class="name">Tri Dao</span>
        </p>
        <p class="affiliation">
        <span class="affiliation">Princeton University</span>
        </p>
      
    </div>
    <div>
      <h3>Published</h3>
      
        <p>April 22, 2026</p> 
    </div>
  </div>
</d-byline> <d-article>
<!-- MathJax (in body so Hugo preserves it) -->
<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$']], displayMath: [['$$','$$']]},
  svg: {fontCache: 'global'}
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<!-- Widen PaperMod content column + hide redundant Distill chrome elements -->
<style id="bilingual-layout-overrides">
/* Override PaperMod's content-width CSS variable + use specificity */
:root, html, body { --main-width: min(1500px, 96vw) !important; --content-gap: 20px !important; }
html body .main,
html body main.main,
html body .post-single,
html body .post-content,
html body article.post-single {
  max-width: min(1500px, 96vw) !important;
  width: min(1500px, 96vw) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
html body .post-content { max-width: 100% !important; padding: 0 4px !important; }
html body .post-header { max-width: 100% !important; }
html body .breadcrumbs { max-width: min(1500px, 96vw) !important; margin: 0 auto !important; }
/* Hide Distill template chrome that doesn't make sense embedded in PaperMod */
.post-content > header,
.post-content #navbar,
.post-content .navbar,
.post-content nav.navbar,
.post-content progress#progress,
.post-content d-title,
.post-content d-front-matter,
.post-content footer.distill-site-footer,
.post-content d-citation-list,
.post-content d-appendix > h3:first-of-type { display: none !important; }
/* d-byline still useful but tame layout */
.post-content d-byline,
.post-content .post .d-byline {
  display: block !important;
  max-width: 100% !important;
  margin: 0 0 24px !important;
  padding: 10px 0 !important;
  border-top: 1px solid var(--border, #ddd);
  border-bottom: 1px solid var(--border, #ddd);
  font-size: 0.85em !important;
}
.post-content d-byline > * {
  display: inline-block !important;
  margin-right: 18px !important;
  vertical-align: top;
}
.post-content d-byline h3 {
  font-size: 0.78em !important;
  text-transform: uppercase;
  color: var(--secondary, #888);
  margin: 0 0 4px !important;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.post-content d-byline p { margin: 0 !important; font-size: 0.95em; }
/* Distill list styling (may render bare bullets) */
.post-content header ul, .post-content nav ul { display: none !important; }
/* Avoid horizontal scroll on the page itself; the prologue handles its own */
body { overflow-x: hidden; }
/* Ensure code/pre wrapping isn't broken by the wider layout */
.post-content pre { white-space: pre; overflow-x: auto; }
/* Tighten d-figure margins */
.post-content d-figure { display: block; margin: 1.4em 0; max-width: 100%; }
.post-content d-figure svg { max-width: 100%; height: auto; }
.post-content d-math { font-size: 1.02em; }
</style>

<div style="background:#fff5f0;border:2px solid #b85450;border-left:5px solid #b85450;padding:14px 20px;margin:16px 0 24px;border-radius:4px;color:#4a1515;font-size:14.5px;line-height:1.7;font-family:-apple-system,sans-serif;">
  <div style="font-weight:700;font-size:15.5px;margin-bottom:6px;color:#721c24">📌 个人学习笔记 · Personal Study Note</div>
  本页是我个人阅读 Dao AI Lab 博客 <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/" style="color:#721c24;font-weight:600" target="_blank">SonicMoE on Blackwell</a>（对应论文 <a href="https://arxiv.org/abs/2512.14080" style="color:#721c24;font-weight:600" target="_blank">arXiv:2512.14080</a>）时做的中英对照 + 深度解读笔记，仅供自己学习备查，<b>不对外分发</b>。原博客正文版权归 Dao AI Lab 所有，所有技术主张、图片、数据归原作者。公开可分享的独立技术解读请见 <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/" style="color:#721c24;font-weight:600" target="_blank">原博客</a>。
</div>

<div class="zh-banner">
  <b>📘 中英对照 + 深度解读版</b><br>
  本文在博客原文（英文）每段之后紧跟 <span style="background:#f0f7ff;padding:1px 5px;border-left:2px solid #4a90e2;">蓝色框中文译文</span>；在关键段落之后插入 <span style="background:#eef7ee;padding:1px 5px;border-left:2px solid #5fa55f;">绿色框深度解读</span>。<br>
  阅读前建议先过 <span style="background:#fff8e7;padding:1px 5px;border-left:2px solid #e0b300;">👇 下方的背景知识与符号定义</span>，后面所有章节都会引用那里的术语与数字。专业术语（TMA/UMMA/TMEM/WGMMA/MMA/CTA/CLC/SMEM/HBM/PTX/CUTLASS/QuACK/SwiGLU/Hopper/Blackwell/Grouped GEMM 等）保留英文避免歧义。
</div>

<section class="prologue" id="prologue">
  <h2 class="prologue-title">📖 Preliminaries · 训练背景知识与符号定义</h2>
  <p class="prologue-intro">本节在阅读正文前铺垫必要的背景，覆盖 <b>MoE 训练流程</b>、<b>NVIDIA GPU 执行 / 内存层级</b>、<b>Tensor Core 指令家族</b>、<b>Hopper vs Blackwell 优化点全景</b>、以及<b>符号速查表</b>。下文所有章节的"深度解读"都会引用这里的术语与数字。</p>

  <div class="prologue-toc">
    <b style="font-size:14px;color:#7a4e00">📑 Prologue 目录</b>
    <ol>
      <li><a href="#pr-s1">① MoE 训练流程回顾</a><span class="toc-sub"> — forward/backward 数据流图 + cache 依赖表 + dS 重排公式 + <a href="#pr-stepwise" style="color:#d6336c;font-weight:700">逐步对照表</a></span></li>
      <li><a href="#pr-s2">② NVIDIA GPU 执行层级</a><span class="toc-sub"> — Grid / Cluster / CTA / Warpgroup / Warp / Thread</span></li>
      <li><a href="#pr-s3">③ 内存层级与带宽（以 B300 为基准）</a><span class="toc-sub"> — Register / SMEM / TMEM / L2 / HBM / NVLink / IB</span></li>
      <li><a href="#pr-s4">④ Tensor Core 指令家族演进</a><span class="toc-sub"> — MMA → WGMMA → UMMA · 数据搬运指令家族</span></li>
      <li><a href="#pr-s5">⑤ 本文涉及的 Hopper / Blackwell 优化点全景</a><span class="toc-new">NEW</span><span class="toc-sub"> — 11 项优化点对照 + 三层叠加图</span></li>
      <li><a href="#pr-s6">⑥ Grouped GEMM / varlen-M / varlen-K</a></li>
      <li><a href="#pr-s7">⑦ 软件栈：CUTLASS / CuTeDSL / QuACK</a></li>
      <li><a href="#pr-s8">⑧ 符号速查表</a><span class="toc-sub"> — $T, d, n, E, K, G, \rho$ · forward/backward 张量 · routing metadata</span></li>
    </ol>
    <p class="toc-tip">📌 读完 Prologue 后再看原博客正文，每一节的"深度解读"都会引用 Prologue 的术语与数字。原博客自己的目录见下方 <b>Contents</b>（由 Distill 模板生成）。</p>
  </div>

  <!-- ========================================= -->
  <h3 class="prologue-h3" id="pr-s1">① MoE 训练流程回顾</h3>
  <p>一个典型 MoE FFN 层对 microbatch 内 $T$ 个 token 做如下处理：</p>
  <ol>
    <li><b>Router</b>：$\text{router\_logits} = X W_r$，$X \in \mathbb{R}^{T \times d}$，$W_r \in \mathbb{R}^{d \times E}$。对每个 token 取 top-K，得到 $K$ 个被激活的 expert 与对应 score $s \in \mathbb{R}^{T \times K}$。</li>
    <li><b>Gather</b>：按 routing 把每个 token 的副本按 expert 排序打包，形成 grouped 输入 $X_g \in \mathbb{R}^{TK \times d}$（每个 token 出现 $K$ 次）。</li>
    <li><b>Up-proj</b>：对每个 expert $e$ 独立做 $H_e = X_{g,e} W_{1,e}^\top$，其中 $W_{1,e} \in \mathbb{R}^{2n \times d}$（gate + up 两半）。合起来是一次 varlen-M Grouped GEMM。</li>
    <li><b>Activation</b>：$A = \mathrm{SwiGLU}(H)$，即 $\mathrm{silu}(H_\text{gate}) \odot H_\text{up}$，输出 $A \in \mathbb{R}^{TK \times n}$。</li>
    <li><b>Down-proj</b>：$Y_e = A_e W_{2,e}^\top$，$W_{2,e} \in \mathbb{R}^{d \times n}$。得到 $Y \in \mathbb{R}^{TK \times d}$。</li>
    <li><b>Scatter + weighted sum</b>：每个 token 把自己的 $K$ 个 expert 输出按 $s$ 加权求和 —— $O_t = \sum_{k=1}^{K} s_{t,k} \cdot Y_{\pi(t,k)}$，输出 $O \in \mathbb{R}^{T \times d}$。</li>
  </ol>
  <p>反向需要：$dO \to dY, dA, dH, dS, dX, dW_1, dW_2$。核心痛点：若按教科书链式法则，中间 $Y, dY$ 都要 materialize 到 HBM，大小 $TKd$ 随 $K$ 线性膨胀 —— 这就是 SonicMoE 要攻破的点。</p>

  <h4 class="prologue-h4">📊 Forward / Backward 数据流与 cache 依赖</h4>

  <!-- ============ SVG: Standard MoE — paper Figure 2 conventions ============ -->
  <p style="font-weight:600;margin-top:12px;color:#7a4e00">Standard MoE — 6 forward kernels + 9 backward kernels （按论文 Figure 2 conventions：黄色=kernel container · 蓝色=intermediate/weight · 红色边框=cached activation · 紫色=output）</p>
  <div class="svg-wrapper">
  <svg viewBox="0 0 1280 740" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#fffefb;border:1px solid #d9b860;border-radius:4px;font-family:-apple-system,sans-serif;">
    <defs>
      <marker id="arrk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker>
      <marker id="arrkr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#888"/></marker>
    </defs>

    <!-- Legend -->
    <g font-size="11">
      <rect x="20" y="14" width="20" height="14" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
      <text x="46" y="25">kernel</text>
      <rect x="100" y="14" width="20" height="14" fill="#dae8fc" stroke="#6c8ebf"/>
      <text x="126" y="25">intermediate / weight</text>
      <rect x="265" y="14" width="20" height="14" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
      <text x="291" y="25" fill="#b85450" font-weight="600">cached for backward</text>
      <rect x="455" y="14" width="20" height="14" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
      <text x="481" y="25" fill="#5a3475" font-weight="600">output</text>
    </g>

    <!-- ============ FORWARD ============ -->
    <text x="20" y="64" font-weight="700" font-size="14" fill="#444">Forward pass · 6 kernels</text>

    <!-- π input above Gather -->
    <text x="135" y="78" text-anchor="middle" font-style="italic" font-size="13" fill="#b85450" font-weight="700">π</text>
    <line x1="135" y1="84" x2="135" y2="100" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- W₁ -->
    <text x="365" y="78" text-anchor="middle" font-style="italic" font-size="13">W₁</text>
    <line x1="365" y1="84" x2="365" y2="100" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- W₂ -->
    <text x="755" y="78" text-anchor="middle" font-style="italic" font-size="13">W₂</text>
    <line x1="755" y1="84" x2="755" y2="100" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- π for scatter -->
    <text x="945" y="78" text-anchor="middle" font-style="italic" font-size="13" fill="#b85450" font-weight="700">π</text>
    <line x1="945" y1="84" x2="945" y2="100" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- π, S for aggregation -->
    <text x="1115" y="78" text-anchor="middle" font-size="13"><tspan font-style="italic" fill="#b85450" font-weight="700">π</tspan>, <tspan font-style="italic" fill="#b85450" font-weight="700">S</tspan></text>
    <line x1="1115" y1="84" x2="1115" y2="100" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- X (input, cached red) -->
    <rect x="10" y="125" width="55" height="40" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
    <text x="38" y="150" text-anchor="middle" font-style="italic" font-size="14" fill="#b85450" font-weight="700">X</text>
    <text x="38" y="183" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">cached</text>
    <line x1="65" y1="145" x2="80" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Gather kernel -->
    <rect x="80" y="100" width="110" height="90" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="135" y="148" text-anchor="middle" font-weight="600" font-size="12">Gather</text>
    <line x1="190" y1="145" x2="205" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- X̃ (cached) -->
    <rect x="205" y="125" width="70" height="40" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
    <text x="240" y="150" text-anchor="middle" font-style="italic" font-size="14" fill="#b85450" font-weight="700">X̃</text>
    <text x="240" y="183" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">cached · [TK,d] 2GB</text>
    <line x1="275" y1="145" x2="290" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Up-proj GEMM kernel -->
    <rect x="290" y="100" width="150" height="90" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="365" y="142" text-anchor="middle" font-weight="600" font-size="12">Up-proj</text>
    <text x="365" y="158" text-anchor="middle" font-size="11" font-style="italic" fill="#7a4e00">Varlen-M Grouped GEMM</text>
    <line x1="440" y1="145" x2="455" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- H (cached) -->
    <rect x="455" y="125" width="60" height="40" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
    <text x="485" y="150" text-anchor="middle" font-style="italic" font-size="14" fill="#b85450" font-weight="700">H</text>
    <text x="485" y="183" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">cached · 1.5GB</text>
    <line x1="515" y1="145" x2="530" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Act func kernel -->
    <rect x="530" y="100" width="100" height="90" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="580" y="148" text-anchor="middle" font-weight="600" font-size="12">Act func</text>
    <line x1="630" y1="145" x2="645" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- A (cached) -->
    <rect x="645" y="125" width="60" height="40" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
    <text x="675" y="150" text-anchor="middle" font-style="italic" font-size="14" fill="#b85450" font-weight="700">A</text>
    <text x="675" y="183" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">cached · 768MB</text>
    <line x1="705" y1="145" x2="680" y2="145" stroke="none"/>
    <line x1="705" y1="145" x2="720" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Down-proj GEMM kernel -->
    <rect x="720" y="100" width="150" height="90" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="795" y="142" text-anchor="middle" font-weight="600" font-size="12">Down-proj</text>
    <text x="795" y="158" text-anchor="middle" font-size="11" font-style="italic" fill="#7a4e00">Varlen-M Grouped GEMM</text>
    <line x1="870" y1="145" x2="885" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Y (cached) -->
    <rect x="885" y="125" width="60" height="40" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
    <text x="915" y="150" text-anchor="middle" font-style="italic" font-size="14" fill="#b85450" font-weight="700">Y</text>
    <text x="915" y="183" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">cached · 2GB ⚠</text>
    <line x1="945" y1="145" x2="960" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Scatter kernel -->
    <rect x="960" y="100" width="100" height="90" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="1010" y="148" text-anchor="middle" font-weight="600" font-size="12">Scatter</text>
    <line x1="1060" y1="145" x2="1075" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Ỹ (cached) -->
    <rect x="1075" y="125" width="60" height="40" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
    <text x="1105" y="150" text-anchor="middle" font-style="italic" font-size="14" fill="#b85450" font-weight="700">Ỹ</text>
    <text x="1105" y="183" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">cached · 2GB</text>
    <line x1="1135" y1="145" x2="1150" y2="145" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Aggregation kernel -->
    <rect x="1150" y="100" width="100" height="90" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="1200" y="142" text-anchor="middle" font-weight="600" font-size="12">Each token</text>
    <text x="1200" y="156" text-anchor="middle" font-size="11" fill="#7a4e00">sums weighted</text>
    <text x="1200" y="170" text-anchor="middle" font-size="11" fill="#7a4e00">expert outputs</text>
    <line x1="1200" y1="190" x2="1200" y2="205" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- O (output, purple) -->
    <rect x="1170" y="205" width="60" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="1200" y="230" text-anchor="middle" font-style="italic" font-size="14" font-weight="700" fill="#5a3475">O</text>
    <text x="1200" y="262" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>

    <!-- Forward summary -->
    <text x="640" y="310" text-anchor="middle" font-size="12" fill="#b85450" font-weight="700">⚠ 5 个 cached O(TKd) 张量：X̃ + H + A + Y + Ỹ ≈ 8.3 GB / 层</text>

    <!-- Divider -->
    <line x1="20" y1="335" x2="1260" y2="335" stroke="#999" stroke-width="0.5" stroke-dasharray="4,4"/>

    <!-- ============ BACKWARD ACT GRAD ============ -->
    <text x="20" y="365" font-weight="700" font-size="14" fill="#444">Backward pass — activation gradient · 6 kernels (right → left)</text>

    <!-- W₂ above down-proj act grad -->
    <text x="755" y="380" text-anchor="middle" font-style="italic" font-size="13">W₂</text>
    <line x1="755" y1="386" x2="755" y2="402" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- W₁ above up-proj act grad -->
    <text x="365" y="380" text-anchor="middle" font-style="italic" font-size="13">W₁</text>
    <line x1="365" y1="386" x2="365" y2="402" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- π,S above aggregation back -->
    <text x="1115" y="380" text-anchor="middle" font-size="13"><tspan font-style="italic" fill="#b85450" font-weight="700">π</tspan>, <tspan font-style="italic" fill="#b85450" font-weight="700">S</tspan></text>
    <line x1="1115" y1="386" x2="1115" y2="402" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- π above scatter back -->
    <text x="945" y="380" text-anchor="middle" font-style="italic" font-size="13" fill="#b85450" font-weight="700">π</text>
    <line x1="945" y1="386" x2="945" y2="402" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- π above gather back -->
    <text x="135" y="380" text-anchor="middle" font-style="italic" font-size="13" fill="#b85450" font-weight="700">π</text>
    <line x1="135" y1="386" x2="135" y2="402" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- dO (input from right) -->
    <rect x="1255" y="425" width="55" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="1282" y="450" text-anchor="middle" font-style="italic" font-size="14">dO</text>
    <line x1="1255" y1="445" x2="1250" y2="445" stroke="none"/>
    <line x1="1255" y1="445" x2="1250" y2="445" stroke="#222" stroke-width="1.3"/>
    <!-- arrow dO → aggregation back (going left) -->
    <line x1="1255" y1="445" x2="1250" y2="445" stroke="#222"/>
    <line x1="1250" y1="445" x2="1265" y2="445" stroke="#222"/>
    <line x1="1250" y1="445" x2="1250" y2="445"/>
    <line x1="1255" y1="445" x2="1250" y2="445" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Aggregation back kernel -->
    <rect x="1150" y="402" width="100" height="80" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="1200" y="437" text-anchor="middle" font-weight="600" font-size="11">Aggregation</text>
    <text x="1200" y="453" text-anchor="middle" font-size="11" font-weight="600">backward</text>
    <line x1="1150" y1="442" x2="1135" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- dỸ -->
    <rect x="1075" y="422" width="60" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="1105" y="447" text-anchor="middle" font-style="italic" font-size="13">dỸ</text>
    <line x1="1075" y1="442" x2="1060" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Scatter back kernel -->
    <rect x="960" y="402" width="100" height="80" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="1010" y="437" text-anchor="middle" font-weight="600" font-size="11">Scatter</text>
    <text x="1010" y="453" text-anchor="middle" font-size="11" font-weight="600">backward</text>
    <line x1="960" y1="442" x2="945" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- dY -->
    <rect x="885" y="422" width="60" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="915" y="447" text-anchor="middle" font-style="italic" font-size="13">dY</text>
    <line x1="885" y1="442" x2="870" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Down-proj act grad kernel -->
    <rect x="720" y="402" width="150" height="80" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="795" y="432" text-anchor="middle" font-weight="600" font-size="11">Down-proj act grad</text>
    <text x="795" y="450" text-anchor="middle" font-size="10.5" font-style="italic" fill="#7a4e00">Varlen-M Grouped GEMM</text>
    <line x1="720" y1="442" x2="705" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- dA -->
    <rect x="645" y="422" width="60" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="675" y="447" text-anchor="middle" font-style="italic" font-size="13">dA</text>
    <line x1="645" y1="442" x2="630" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- dAct func kernel (uses cached H) -->
    <rect x="530" y="402" width="100" height="80" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="580" y="432" text-anchor="middle" font-weight="600" font-size="11">dAct func</text>
    <text x="580" y="450" text-anchor="middle" font-size="9" fill="#b85450">uses cached H</text>
    <line x1="530" y1="442" x2="515" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- Cache arrow from H → dAct -->
    <line x1="485" y1="165" x2="485" y2="200" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="485" y1="200" x2="565" y2="200" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="565" y1="200" x2="565" y2="402" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none" marker-end="url(#arrkr)"/>

    <!-- dH -->
    <rect x="455" y="422" width="60" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="485" y="447" text-anchor="middle" font-style="italic" font-size="13">dH</text>
    <line x1="455" y1="442" x2="440" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Up-proj act grad kernel -->
    <rect x="290" y="402" width="150" height="80" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="365" y="432" text-anchor="middle" font-weight="600" font-size="11">Up-proj act grad</text>
    <text x="365" y="450" text-anchor="middle" font-size="10.5" font-style="italic" fill="#7a4e00">Varlen-M Grouped GEMM</text>
    <line x1="290" y1="442" x2="275" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- dX̃ -->
    <rect x="205" y="422" width="70" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="240" y="447" text-anchor="middle" font-style="italic" font-size="13">dX̃</text>
    <line x1="205" y1="442" x2="190" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- Gather back / Aggregation kernel -->
    <rect x="80" y="402" width="110" height="80" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="135" y="432" text-anchor="middle" font-weight="600" font-size="11">Aggregation</text>
    <text x="135" y="450" text-anchor="middle" font-size="11" font-weight="600">(gather back)</text>
    <line x1="80" y1="442" x2="65" y2="442" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>

    <!-- dX (output, purple) -->
    <rect x="10" y="422" width="55" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="38" y="447" text-anchor="middle" font-style="italic" font-size="13" font-weight="700" fill="#5a3475">dX</text>
    <text x="38" y="478" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>

    <!-- Divider -->
    <line x1="20" y1="510" x2="1260" y2="510" stroke="#999" stroke-width="0.5" stroke-dasharray="4,4"/>

    <!-- ============ BACKWARD WEIGHT GRAD + dS ============ -->
    <text x="20" y="540" font-weight="700" font-size="14" fill="#444">Backward pass — weight gradient + dS · 3 kernels（每个都依赖一个 cached O(TKd) 张量）</text>

    <!-- dW₁ kernel: uses dH + cached X̃ -->
    <rect x="180" y="565" width="170" height="80" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="265" y="595" text-anchor="middle" font-weight="600" font-size="11">Up-proj weight grad</text>
    <text x="265" y="613" text-anchor="middle" font-size="10.5" font-style="italic" fill="#7a4e00">Varlen-K Grouped GEMM</text>
    <text x="265" y="630" text-anchor="middle" font-size="9" fill="#b85450" font-weight="600">需 cached X̃ + dH</text>
    <!-- arrow down to output -->
    <line x1="265" y1="645" x2="265" y2="660" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- input from dH -->
    <line x1="485" y1="462" x2="485" y2="540" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="485" y1="540" x2="305" y2="540" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="305" y1="540" x2="305" y2="565" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none" marker-end="url(#arrkr)"/>
    <!-- input from cached X̃ -->
    <line x1="240" y1="165" x2="240" y2="220" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="240" y1="220" x2="240" y2="565" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none" marker-end="url(#arrkr)"/>

    <!-- dW₁ output -->
    <rect x="225" y="660" width="80" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="265" y="685" text-anchor="middle" font-style="italic" font-size="14" font-weight="700" fill="#5a3475">dW₁</text>
    <text x="265" y="717" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>

    <!-- dW₂ kernel: uses dY + cached A -->
    <rect x="610" y="565" width="170" height="80" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="695" y="595" text-anchor="middle" font-weight="600" font-size="11">Down-proj weight grad</text>
    <text x="695" y="613" text-anchor="middle" font-size="10.5" font-style="italic" fill="#7a4e00">Varlen-K Grouped GEMM</text>
    <text x="695" y="630" text-anchor="middle" font-size="9" fill="#b85450" font-weight="600">需 cached A + dY</text>
    <line x1="695" y1="645" x2="695" y2="660" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- input from dY -->
    <line x1="915" y1="462" x2="915" y2="540" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="915" y1="540" x2="735" y2="540" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="735" y1="540" x2="735" y2="565" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none" marker-end="url(#arrkr)"/>
    <!-- input from cached A -->
    <line x1="675" y1="165" x2="675" y2="220" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="675" y1="220" x2="675" y2="565" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none" marker-end="url(#arrkr)"/>

    <!-- dW₂ output -->
    <rect x="655" y="660" width="80" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="695" y="685" text-anchor="middle" font-style="italic" font-size="14" font-weight="700" fill="#5a3475">dW₂</text>
    <text x="695" y="717" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>

    <!-- dS kernel: uses dO + cached Y (the painful one!) -->
    <rect x="1010" y="565" width="180" height="80" rx="6" fill="#fff8d5" stroke="#b85450" stroke-width="2.5"/>
    <text x="1100" y="595" text-anchor="middle" font-weight="700" font-size="11" fill="#b85450">dS = ⟨dO, Y⟩</text>
    <text x="1100" y="613" text-anchor="middle" font-size="10.5" font-style="italic" fill="#b85450">row-wise inner product</text>
    <text x="1100" y="630" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">⚠ MUST cache Y (2GB)</text>
    <line x1="1100" y1="645" x2="1100" y2="660" stroke="#222" stroke-width="1.3" marker-end="url(#arrk)"/>
    <!-- input from cached Y -->
    <line x1="915" y1="165" x2="915" y2="220" stroke="#b85450" stroke-width="1.5" stroke-dasharray="4,3" fill="none"/>
    <line x1="915" y1="220" x2="1015" y2="220" stroke="#b85450" stroke-width="1.5" stroke-dasharray="4,3" fill="none"/>
    <line x1="1015" y1="220" x2="1015" y2="565" stroke="#b85450" stroke-width="1.5" stroke-dasharray="4,3" fill="none" marker-end="url(#arrkr)"/>
    <!-- input from dO -->
    <line x1="1282" y1="465" x2="1282" y2="540" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="1282" y1="540" x2="1185" y2="540" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none"/>
    <line x1="1185" y1="540" x2="1185" y2="565" stroke="#888" stroke-width="1" stroke-dasharray="3,3" fill="none" marker-end="url(#arrkr)"/>

    <!-- dS output -->
    <rect x="1060" y="660" width="80" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="1100" y="685" text-anchor="middle" font-style="italic" font-size="14" font-weight="700" fill="#5a3475">dS</text>
    <text x="1100" y="717" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>
  </svg>
  </div>
  <p style="font-size:12px;color:#7a4e00;margin-top:6px"><b>读图：</b>实线 = kernel 之间数据流；灰色虚线 = 反向 kernel 依赖某个 cached forward 张量；红色虚线 = SonicMoE 要消灭的关键依赖（dS 必须 cache Y）。每个红色边框 blue box 是一个被 cache 的 O(TKd) activation；红色 π/S 标记表明这两个也是 cached（小张量，与 K 无关，无所谓）。</p>

  <!-- Standard formulas -->
  <div class="formula-box std-box">
    <div class="formula-label">Standard MoE 的前向 / 反向公式（教科书链式法则直写）</div>

    <p style="margin:6px 0 2px"><b>前向</b>（对 token $t$，激活 $K$ 个 expert）：</p>
    <ul class="fml-list">
      <li>up-proj：$H_{e,t} = X_t \cdot W_{1,e}^{\mathsf T}$</li>
      <li>activation：$A_{e,t} = \mathrm{SwiGLU}(H_{e,t})$</li>
      <li>down-proj：$Y_{e,t} = A_{e,t} \cdot W_{2,e}$</li>
      <li>aggregate：$O_t = \sum_{k=1}^{K} s_{t,k} \cdot Y_{e_k, t}$</li>
    </ul>

    <p style="margin:10px 0 2px"><b>反向</b>（必须先 materialize $dY$）：</p>
    <table class="fml-tbl std">
      <tr><td class="fml-eq">$dY_{e,t} = s_{t,e} \cdot dO_t$</td><td class="fml-note">⚠ materialize $[TK,d]$ = 2 GB</td></tr>
      <tr><td class="fml-eq">$dA_{e,t} = dY_{e,t} \cdot W_{2,e}^{\mathsf T} = s_{t,e} \cdot dA'_{e,t}$，其中 $dA'_{e,t} := dO_t \cdot W_{2,e}^{\mathsf T}$</td><td class="fml-note">中间张量</td></tr>
      <tr><td class="fml-eq">$dH_{e,t} = dA_{e,t} \odot J_{\mathrm{SwiGLU}}(H_{e,t})$</td><td class="fml-note">⚠ 需 cached $H$</td></tr>
      <tr><td class="fml-eq">$dW_{2,e} = \sum_t A_{e,t}^{\mathsf T} \cdot dY_{e,t}$</td><td class="fml-note">⚠ 需 cached $A$</td></tr>
      <tr><td class="fml-eq">$dS_{t,e} = \langle dO_t,\; Y_{e,t} \rangle$</td><td class="fml-note">⚠ 需 cached $Y$</td></tr>
      <tr><td class="fml-eq">$dW_{1,e} = \sum_t X_{g,e,t}^{\mathsf T} \cdot dH_{e,t}$</td><td class="fml-note">⚠ 需 cached $X_g$</td></tr>
    </table>

    <p style="margin:8px 0 0;color:#721c24;font-size:13px"><b>结论：</b>$A, Y, X_g$ 必须 cache 给反向用；加上 $H$ 与 $dY$ materialize，一共 $O(TKd)$ 量级张量 ×4。</p>
  </div>

  <!-- ============ SVG: SonicMoE — paper Figure 3 conventions ============ -->
  <p style="font-weight:600;margin-top:18px;color:#1a3d1a">SonicMoE — 3 forward kernels + 5 backward kernels（严格按论文 Figure 3 conventions：黄色=kernel · 蓝色=intermediate/weight · 红色=cached（X, H, π, S） · 紫色=output（O, dX, dW₁, dW₂））</p>
  <div class="svg-wrapper">
  <svg viewBox="0 0 1280 740" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#fafffa;border:1px solid #5fa55f;border-radius:4px;font-family:-apple-system,sans-serif;">
    <defs>
      <marker id="arrs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker>
      <marker id="arrsr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#888"/></marker>
    </defs>

    <!-- Legend -->
    <g font-size="11">
      <rect x="20" y="14" width="20" height="14" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
      <text x="46" y="25">kernel</text>
      <rect x="100" y="14" width="20" height="14" fill="#dae8fc" stroke="#6c8ebf"/>
      <text x="126" y="25">intermediate / weight</text>
      <rect x="265" y="14" width="20" height="14" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
      <text x="291" y="25" fill="#b85450" font-weight="600">cached for backward</text>
      <rect x="455" y="14" width="20" height="14" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
      <text x="481" y="25" fill="#5a3475" font-weight="600">output</text>
    </g>

    <!-- ============ FORWARD ============ -->
    <text x="20" y="64" font-weight="700" font-size="14" fill="#444">Forward pass · 3 kernels</text>

    <!-- Inputs above kernels -->
    <text x="290" y="78" text-anchor="middle" font-size="13"><tspan font-style="italic">W₁</tspan>, <tspan font-style="italic" fill="#b85450" font-weight="700">π</tspan></text>
    <line x1="290" y1="84" x2="290" y2="100" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>
    <text x="700" y="78" text-anchor="middle" font-style="italic" font-size="13">W₂</text>
    <line x1="700" y1="84" x2="700" y2="100" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>
    <text x="1090" y="78" text-anchor="middle" font-size="13"><tspan font-style="italic" fill="#b85450" font-weight="700">π</tspan>, <tspan font-style="italic" fill="#b85450" font-weight="700">S</tspan></text>
    <line x1="1090" y1="84" x2="1090" y2="100" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- X (cached input) -->
    <rect x="40" y="135" width="60" height="40" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
    <text x="70" y="160" text-anchor="middle" font-style="italic" font-size="14" fill="#b85450" font-weight="700">X</text>
    <text x="70" y="193" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">cached · [T,d]</text>
    <line x1="100" y1="155" x2="115" y2="155" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- Up-proj kernel (large yellow box containing GEMM + Act func) -->
    <rect x="115" y="100" width="350" height="125" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="290" y="120" text-anchor="middle" font-weight="700" font-size="12" fill="#7a4e00">Up-proj</text>
    <!-- Inner: Varlen-M Grouped GEMM box -->
    <rect x="135" y="135" width="120" height="50" fill="#ffffff" stroke="#7a4e00" stroke-width="0.8"/>
    <text x="195" y="156" text-anchor="middle" font-size="10.5" font-style="italic">Varlen-M</text>
    <text x="195" y="170" text-anchor="middle" font-size="10.5" font-style="italic">Grouped GEMM</text>
    <line x1="255" y1="160" x2="280" y2="160" stroke="#222" stroke-width="1.2" marker-end="url(#arrs)"/>
    <!-- H output of inner GEMM, cached -->
    <rect x="280" y="140" width="50" height="40" fill="#dae8fc" stroke="#b85450" stroke-width="2"/>
    <text x="305" y="165" text-anchor="middle" font-style="italic" font-size="13" fill="#b85450" font-weight="700">H</text>
    <line x1="330" y1="160" x2="350" y2="160" stroke="#222" stroke-width="1.2" marker-end="url(#arrs)"/>
    <!-- Inner: Act func box -->
    <rect x="350" y="135" width="100" height="50" fill="#ffffff" stroke="#7a4e00" stroke-width="0.8"/>
    <text x="400" y="165" text-anchor="middle" font-size="11" font-style="italic">Act func</text>
    <!-- H cached marker below -->
    <text x="305" y="200" text-anchor="middle" font-size="9" fill="#b85450" font-weight="700">cached · 1.5GB ★</text>

    <!-- arrow Up-proj kernel → A intermediate -->
    <line x1="465" y1="155" x2="490" y2="155" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- A intermediate -->
    <rect x="490" y="135" width="55" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="518" y="160" text-anchor="middle" font-style="italic" font-size="14">A</text>
    <line x1="545" y1="155" x2="565" y2="155" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- Down-proj kernel -->
    <rect x="565" y="100" width="270" height="125" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="700" y="120" text-anchor="middle" font-weight="700" font-size="12" fill="#7a4e00">Down-proj</text>
    <!-- Inner: Varlen-M Grouped GEMM -->
    <rect x="600" y="135" width="200" height="50" fill="#ffffff" stroke="#7a4e00" stroke-width="0.8"/>
    <text x="700" y="156" text-anchor="middle" font-size="10.5" font-style="italic">Varlen-M</text>
    <text x="700" y="170" text-anchor="middle" font-size="10.5" font-style="italic">Grouped GEMM</text>

    <line x1="835" y1="155" x2="855" y2="155" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- Y intermediate -->
    <rect x="855" y="135" width="55" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="883" y="160" text-anchor="middle" font-style="italic" font-size="14">Y</text>
    <line x1="910" y1="155" x2="930" y2="155" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- Expert aggregation kernel -->
    <rect x="930" y="100" width="290" height="125" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="1075" y="120" text-anchor="middle" font-weight="700" font-size="12" fill="#7a4e00">Expert aggregation</text>
    <rect x="950" y="135" width="250" height="50" fill="#ffffff" stroke="#7a4e00" stroke-width="0.8"/>
    <text x="1075" y="156" text-anchor="middle" font-size="10.5" font-style="italic">Each token gathers</text>
    <text x="1075" y="170" text-anchor="middle" font-size="10.5" font-style="italic">and sums expert outputs</text>

    <line x1="1220" y1="155" x2="1240" y2="155" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- O (output, purple) -->
    <rect x="1240" y="135" width="35" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="1257" y="160" text-anchor="middle" font-style="italic" font-size="14" font-weight="700" fill="#5a3475">O</text>
    <text x="1257" y="193" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>

    <!-- Forward summary -->
    <text x="640" y="280" text-anchor="middle" font-size="12" fill="#1f5d1f" font-weight="700">✓ 仅 X 与 H 被 cache（1.8 GB / 层）；A、Y 在 kernel 内 ephemeral，不写 HBM</text>

    <!-- Divider -->
    <line x1="20" y1="305" x2="1260" y2="305" stroke="#999" stroke-width="0.5" stroke-dasharray="4,4"/>

    <!-- ============ BACKWARD ============ -->
    <text x="20" y="335" font-weight="700" font-size="14" fill="#444">Backward pass · 5 kernels</text>

    <!-- Inputs above -->
    <text x="985" y="350" text-anchor="middle" font-size="13"><tspan font-style="italic">W₂</tspan>, <tspan font-style="italic" fill="#b85450" font-weight="700">H</tspan></text>
    <text x="985" y="368" text-anchor="middle" font-size="9" fill="#b85450">(cached H)</text>
    <line x1="985" y1="372" x2="985" y2="388" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>
    <text x="495" y="350" text-anchor="middle" font-style="italic" font-size="13">W₁</text>
    <line x1="495" y1="356" x2="495" y2="388" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>
    <text x="180" y="350" text-anchor="middle" font-style="italic" font-size="13" fill="#b85450" font-weight="700">π</text>
    <line x1="180" y1="356" x2="180" y2="388" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- dO input on right -->
    <rect x="1240" y="408" width="35" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="1257" y="433" text-anchor="middle" font-style="italic" font-size="13">dO</text>
    <line x1="1240" y1="428" x2="1230" y2="428" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- ===== Down-proj act grad kernel (the dH kernel — gemm_dgated) ===== -->
    <rect x="850" y="388" width="380" height="200" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="2"/>
    <text x="1040" y="408" text-anchor="middle" font-weight="700" font-size="12" fill="#7a4e00">Down-proj act grad — gemm_dgated (fused)</text>

    <!-- Inner: Varlen-M Grouped GEMM (dA') -->
    <rect x="1080" y="425" width="135" height="50" fill="#ffffff" stroke="#7a4e00" stroke-width="0.8"/>
    <text x="1147" y="446" text-anchor="middle" font-size="10.5" font-style="italic">Varlen-M</text>
    <text x="1147" y="460" text-anchor="middle" font-size="10.5" font-style="italic">Grouped GEMM</text>
    <!-- dA' produced -->
    <line x1="1080" y1="450" x2="1060" y2="450" stroke="#222" stroke-width="1.2" marker-end="url(#arrs)"/>
    <rect x="1010" y="430" width="50" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="1035" y="455" text-anchor="middle" font-style="italic" font-size="13">dA′</text>

    <!-- dAct func -->
    <rect x="900" y="425" width="100" height="50" fill="#ffffff" stroke="#7a4e00" stroke-width="0.8"/>
    <text x="950" y="455" text-anchor="middle" font-size="11" font-style="italic">dAct func</text>
    <line x1="1010" y1="450" x2="1000" y2="450" stroke="#222" stroke-width="1.2" marker-end="url(#arrs)"/>

    <!-- A (recomputed, sum over n) -->
    <rect x="1010" y="490" width="50" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="1035" y="515" text-anchor="middle" font-style="italic" font-size="13">A</text>
    <text x="1035" y="544" text-anchor="middle" font-size="9" fill="#666">sum over n</text>
    <!-- arrow from dA' (top) and H input down to A -->
    <line x1="1035" y1="470" x2="1035" y2="490" stroke="#222" stroke-width="1" stroke-dasharray="2,2"/>

    <!-- A' = s · A -->
    <rect x="1085" y="490" width="50" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="1110" y="515" text-anchor="middle" font-style="italic" font-size="13">A′</text>
    <line x1="1060" y1="510" x2="1085" y2="510" stroke="#222" stroke-width="1.2" marker-end="url(#arrs)"/>

    <!-- ds path -->
    <rect x="1160" y="490" width="55" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="1187" y="515" text-anchor="middle" font-style="italic" font-size="13">dS</text>
    <text x="1187" y="544" text-anchor="middle" font-size="9" fill="#666">⟨dA′, A⟩</text>

    <!-- arrows for dA' → A and dA' → dS -->
    <line x1="1035" y1="470" x2="1035" y2="490" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
    <line x1="1060" y1="450" x2="1187" y2="490" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2" marker-end="url(#arrsr)"/>

    <!-- S input below for A' = s·A -->
    <text x="1110" y="568" text-anchor="middle" font-size="11" font-style="italic" fill="#b85450" font-weight="700">S</text>
    <line x1="1110" y1="570" x2="1110" y2="535" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>

    <!-- dH output of kernel -->
    <line x1="900" y1="450" x2="850" y2="450" stroke="#222" stroke-width="1.5"/>
    <line x1="850" y1="450" x2="800" y2="450" stroke="#222" stroke-width="1.5" marker-end="url(#arrs)"/>

    <!-- dH intermediate -->
    <rect x="765" y="430" width="55" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="793" y="455" text-anchor="middle" font-style="italic" font-size="13">dH</text>
    <line x1="765" y1="450" x2="700" y2="450" stroke="#222" stroke-width="1.5" marker-end="url(#arrs)"/>

    <!-- ===== Up-proj act grad kernel ===== -->
    <rect x="380" y="388" width="320" height="120" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="540" y="408" text-anchor="middle" font-weight="700" font-size="12" fill="#7a4e00">Up-proj act grad</text>
    <rect x="430" y="425" width="220" height="50" fill="#ffffff" stroke="#7a4e00" stroke-width="0.8"/>
    <text x="540" y="446" text-anchor="middle" font-size="10.5" font-style="italic">Varlen-M</text>
    <text x="540" y="460" text-anchor="middle" font-size="10.5" font-style="italic">Grouped GEMM</text>

    <line x1="380" y1="450" x2="350" y2="450" stroke="#222" stroke-width="1.5" marker-end="url(#arrs)"/>

    <!-- dX̃ -->
    <rect x="295" y="430" width="55" height="40" fill="#dae8fc" stroke="#6c8ebf"/>
    <text x="322" y="455" text-anchor="middle" font-style="italic" font-size="13">dX̃</text>
    <line x1="295" y1="450" x2="270" y2="450" stroke="#222" stroke-width="1.5" marker-end="url(#arrs)"/>

    <!-- ===== Expert aggregation backward ===== -->
    <rect x="100" y="388" width="170" height="120" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="185" y="408" text-anchor="middle" font-weight="700" font-size="12" fill="#7a4e00">Expert aggregation</text>
    <rect x="115" y="425" width="140" height="50" fill="#ffffff" stroke="#7a4e00" stroke-width="0.8"/>
    <text x="185" y="446" text-anchor="middle" font-size="10.5" font-style="italic">Each token gathers</text>
    <text x="185" y="460" text-anchor="middle" font-size="10.5" font-style="italic">and sums</text>

    <line x1="100" y1="450" x2="80" y2="450" stroke="#222" stroke-width="1.5" marker-end="url(#arrs)"/>

    <!-- dX (purple output) -->
    <rect x="40" y="430" width="40" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="60" y="455" text-anchor="middle" font-style="italic" font-size="13" font-weight="700" fill="#5a3475">dX</text>
    <text x="60" y="488" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>

    <!-- Divider -->
    <line x1="20" y1="608" x2="1260" y2="608" stroke="#999" stroke-width="0.5" stroke-dasharray="4,4"/>

    <!-- ============ Backward weight grad ============ -->
    <text x="20" y="635" font-weight="700" font-size="14" fill="#444">Backward weight grad · 2 kernels（A′ 直接来自 dH kernel，无需额外 cache）</text>

    <!-- Up-proj weight grad kernel -->
    <rect x="320" y="650" width="200" height="65" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="420" y="672" text-anchor="middle" font-weight="700" font-size="11" fill="#7a4e00">Up-proj weight grad</text>
    <text x="420" y="690" text-anchor="middle" font-size="10.5" font-style="italic">Varlen-K Grouped GEMM</text>
    <text x="420" y="708" text-anchor="middle" font-size="9" fill="#1f5d1f">cached X (256MB) + dH</text>

    <!-- inputs to up-proj weight grad: π, X (cached), dH -->
    <text x="270" y="640" font-size="11"><tspan font-style="italic" fill="#b85450" font-weight="700">π</tspan>, <tspan font-style="italic" fill="#b85450" font-weight="700">X</tspan>, <tspan font-style="italic">dH</tspan></text>
    <line x1="295" y1="643" x2="320" y2="675" stroke="#222" stroke-width="1.2" marker-end="url(#arrs)"/>

    <!-- dW₁ output -->
    <line x1="520" y1="685" x2="540" y2="685" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>
    <rect x="540" y="665" width="60" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="570" y="690" text-anchor="middle" font-style="italic" font-size="14" font-weight="700" fill="#5a3475">dW₁</text>
    <text x="570" y="722" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>

    <!-- Down-proj weight grad kernel -->
    <rect x="780" y="650" width="220" height="65" rx="6" fill="#fff8d5" stroke="#e0b300" stroke-width="1.5"/>
    <text x="890" y="672" text-anchor="middle" font-weight="700" font-size="11" fill="#7a4e00">Down-proj weight grad</text>
    <text x="890" y="690" text-anchor="middle" font-size="10.5" font-style="italic">Varlen-K Grouped GEMM</text>
    <text x="890" y="708" text-anchor="middle" font-size="9" fill="#1f5d1f">用 A′（来自 dH kernel）+ dO</text>

    <!-- inputs: dO, A', π -->
    <text x="730" y="640" font-size="11"><tspan font-style="italic">dO</tspan>, <tspan font-style="italic">A′</tspan>, <tspan font-style="italic" fill="#b85450" font-weight="700">π</tspan></text>
    <line x1="760" y1="643" x2="780" y2="675" stroke="#222" stroke-width="1.2" marker-end="url(#arrs)"/>

    <!-- dW₂ output -->
    <line x1="1000" y1="685" x2="1020" y2="685" stroke="#222" stroke-width="1.3" marker-end="url(#arrs)"/>
    <rect x="1020" y="665" width="60" height="40" fill="#e1d5e7" stroke="#9673a6" stroke-width="2"/>
    <text x="1050" y="690" text-anchor="middle" font-style="italic" font-size="14" font-weight="700" fill="#5a3475">dW₂</text>
    <text x="1050" y="722" text-anchor="middle" font-size="9" fill="#5a3475" font-weight="700">output</text>

    <!-- Footer -->
    <text x="640" y="730" text-anchor="middle" font-size="12" fill="#1f5d1f" font-weight="700">合计 8 个 launched kernel（forward 3 + backward 5）；activation memory 与 K 解耦</text>
  </svg>
  </div>
  <p style="font-size:12px;color:#1f5d1f;margin-top:6px"><b>读图：</b>这张图严格对应论文 Figure 3。<b>关键差异</b>：dH kernel 的 yellow 容器内同时包含 Varlen-M Grouped GEMM (产 dA′)、dAct func (产 dH)、"sum over n" (重算 A)、以及通过 S/π 的加权得到 A′ —— 一发 kernel 同时输出 dH、A′、dS 三个张量。<b>没有任何 O(TKd) 中间张量需要 materialize 到 HBM</b>，A′ 的"写 HBM"虽然看起来仍是 [TK, n]，但它只是 forward 缓存项 H 的同等代价（4·T·I，与 K 无关），不属于 O(TKd) 类张量。</p>

  <!-- SonicMoE formulas -->
  <div class="formula-box sm-box">
    <div class="formula-label">SonicMoE 的前向 / 反向公式（换序 + <code>colvec_scale</code> 技巧）</div>

    <p style="margin:6px 0 2px"><b>前向</b>（与 Standard 完全相同的语义，只是不 save $X_g, A, Y$）：</p>
    <ul class="fml-list">
      <li>up-proj：$h_{e,t} = X_t \cdot W_{1,e}^{\mathsf T}$（同 Standard 的 $H$）</li>
      <li>activation：$a_{e,t} = \mathrm{SwiGLU}(h_{e,t})$ —— 标 <code>mark_non_differentiable</code>，不进 autograd</li>
      <li>down-proj：$y_{e,t} = a_{e,t} \cdot W_{2,e}$ —— ephemeral，立刻被 aggregation 消费</li>
      <li>aggregate：$O_t = \sum_{k=1}^{K} s_{t,k} \cdot y_{e_k, t}$（数学上与 Standard 完全相同）</li>
    </ul>

    <p style="margin:10px 0 2px"><b>反向</b>（单个 <code>gemm_dgated</code> kernel 同时吐三个输出）：</p>

    <p style="margin:6px 0">mainloop 算：$dA'_{e,t} = dO_t \cdot W_{2,e}^{\mathsf T}$ —— 结果留在 TMEM，<b>从不写 HBM</b>。</p>

    <p style="margin:6px 0">epilogue 内，<code>colvec_scale</code> = $s$ 只作用在 <code>dx_out</code> 和 <code>postact_out</code>，而 <code>colvec_reduce</code> <b>在 scale 之前</b>捕获未 scale 的 $A$：</p>

    <table class="fml-tbl sm">
      <tr><td class="fml-eq"><b>dx_out</b>：&nbsp; $dH_{e,t} = s_{t,e} \cdot \bigl(dA'_{e,t} \odot J_{\mathrm{SwiGLU}}(h_{e,t})\bigr)$</td><td class="fml-note">输出 #1（写 HBM）</td></tr>
      <tr><td class="fml-eq"><b>postact_out</b>：&nbsp; $A'_{e,t} = s_{t,e} \cdot \mathrm{SwiGLU}(h_{e,t}) = s_{t,e} \cdot A_{e,t}$</td><td class="fml-note">输出 #2（写 HBM，喂 $dW_2$）</td></tr>
      <tr><td class="fml-eq"><b>ds_scattered</b>：&nbsp; $dS_{t,e} = \langle dA'_{e,t},\; A_{e,t} \rangle$（$A$ <b>未 scale</b>）</td><td class="fml-note">输出 #3（行归约）</td></tr>
    </table>

    <p style="margin:8px 0">然后两个 varlen-K GEMM 直接用上面的输出：</p>
    <ul class="fml-list">
      <li>$dW_{2,e} = \sum_t dO_t^{\mathsf T} \cdot A'_{e,t} = \sum_t dO_t^{\mathsf T} \cdot (s_{t,e} \cdot A_{e,t})$ —— 单次 <code>gemm</code></li>
      <li>$dW_{1,e} = \sum_t X_t^{\mathsf T} \cdot dH_{e,t}$ —— 单次 <code>gemm</code>，<b>TMA gather4(X) 内联</b></li>
    </ul>

    <p style="margin:8px 0 0;color:#1f3d1f;font-size:13px"><b>结论：</b>反向只读 $h$ 与原始 $X/dO$；$Y, dY, A, X_g$ 全不存在于 HBM。</p>
  </div>

  <!-- ============ Equivalence proof ============ -->
  <div class="eq-box">
    <div class="eq-label">🔎 等价性证明：两种方案 bit-exact 产生相同梯度</div>
    <p style="margin:6px 0 2px"><b>关键观察 1：</b>SonicMoE 的"前向"其实<u>语义上和 Standard 完全一致</u>。$a, y$ 这些中间量在 HBM 里的"生存时间"不同（ephemeral vs cached），但输出 $O$ 逐比特相同。$a$ 在 autograd 层用 <code>mark_non_differentiable</code> 切断，但反向链式法则不走 $a$ 这条边，走的是 $h$ + 重算 —— 数学上等价。</p>

    <p style="margin:12px 0 4px"><b>关键观察 2：三个关键反向量的 bit-exact 等价推导</b></p>

    <div class="eq-step">
      <div class="eq-step-title">① $dS_{t,e}$ 等价（<span style="color:#b85450">Standard</span> vs <span style="color:#1f5d1f">SonicMoE</span>）—— 核心重排</div>
      <table class="fml-tbl derive">
        <tr><td class="fml-src std">Standard 起点</td><td class="fml-eq">$dS_{t,e} = \langle dO_t,\; Y_{e,t} \rangle$</td><td class="fml-note">—</td></tr>
        <tr><td class="fml-src std">代入 $Y = A \cdot W_2$</td><td class="fml-eq">$= \langle dO_t,\; A_{e,t} \cdot W_{2,e} \rangle$</td><td class="fml-note">—</td></tr>
        <tr><td class="fml-src std">展开两重求和</td><td class="fml-eq">$= \sum_{d,i}\, dO_{t,d} \cdot A_{e,t,i} \cdot W_{2,e,i,d}$</td><td class="fml-note">$d$: hidden，$i$: intermediate</td></tr>
        <tr><td class="fml-src std">交换求和顺序</td><td class="fml-eq">$= \sum_{i}\, A_{e,t,i} \cdot \bigl(\sum_{d}\, dO_{t,d} \cdot W_{2,e,i,d}\bigr)$</td><td class="fml-note">合法：有限和</td></tr>
        <tr><td class="fml-src std">识别内层 = $dA'$</td><td class="fml-eq">$= \sum_{i}\, A_{e,t,i} \cdot (dO_t \cdot W_{2,e}^{\mathsf T})_i$</td><td class="fml-note">内层即 $dA'_{e,t,i}$</td></tr>
        <tr><td class="fml-src sm">SonicMoE 直出</td><td class="fml-eq">$= \langle dA'_{e,t},\; A_{e,t} \rangle$ &nbsp;&nbsp; ✓</td><td class="fml-note">epilogue <code>colvec_reduce</code></td></tr>
      </table>
      <p class="eq-concl">⇒ 逐浮点位相同（reduction 维度都是 $i$，只是把乘法凑对的方式不同）。实际实现里 SonicMoE 的 <code>colvec_reduce</code> 在 FP32 register 上做 $I$ 维 row-sum，与 Standard 的 $\sum_d dO_{t,d} \cdot Y_{e,t,d}$ 都是 FP32 accumulation，精度等价。</p>
    </div>

    <div class="eq-step">
      <div class="eq-step-title">② $dW_{2,e}$ 等价</div>
      <table class="fml-tbl derive">
        <tr><td class="fml-src std">Standard 反向 GEMM</td><td class="fml-eq">$dW_{2,e} = \sum_t\, A_{e,t}^{\mathsf T} \cdot dY_{e,t}$</td><td class="fml-note">—</td></tr>
        <tr><td class="fml-src std">代入 $dY = s \cdot dO$</td><td class="fml-eq">$= \sum_t\, A_{e,t}^{\mathsf T} \cdot (s_{t,e} \cdot dO_t)$</td><td class="fml-note">—</td></tr>
        <tr><td class="fml-src std">标量可任意挪</td><td class="fml-eq">$= \sum_t\, (s_{t,e} \cdot A_{e,t})^{\mathsf T} \cdot dO_t$</td><td class="fml-note">把 $s$ 推进转置左</td></tr>
        <tr><td class="fml-src sm">SonicMoE 实现</td><td class="fml-eq">$= \sum_t\, (A'_{e,t})^{\mathsf T} \cdot dO_t$ &nbsp;&nbsp; ✓</td><td class="fml-note"><code>gemm(dout.T, a_prime)</code></td></tr>
        <tr><td class="fml-src sm">代入 $A' = s \cdot A$</td><td class="fml-eq">$= \sum_t\, (s_{t,e} \cdot A_{e,t})^{\mathsf T} \cdot dO_t$</td><td class="fml-note">与 Standard 第 3 行字面相同</td></tr>
      </table>
      <p class="eq-concl">⇒ 两式逐项相同。关键是 <code>postact_out</code> 存的是 $s \cdot A$ 而不是 $A$ —— <code>colvec_scale</code> 在 epilogue 里把 scale 并进 $A'$。</p>
    </div>

    <div class="eq-step">
      <div class="eq-step-title">③ $dH_{e,t}$ 等价</div>
      <table class="fml-tbl derive">
        <tr><td class="fml-src std">Standard 链式</td><td class="fml-eq">$dH_{e,t} = dA_{e,t} \odot J_{\mathrm{SwiGLU}}(H_{e,t})$</td><td class="fml-note">—</td></tr>
        <tr><td class="fml-src std">代入 $dA = dY \cdot W_2^{\mathsf T}$</td><td class="fml-eq">$= (dY_{e,t} \cdot W_{2,e}^{\mathsf T}) \odot J(H)$</td><td class="fml-note">—</td></tr>
        <tr><td class="fml-src std">代入 $dY = s \cdot dO$</td><td class="fml-eq">$= (s_{t,e} \cdot dO_t \cdot W_{2,e}^{\mathsf T}) \odot J(H)$</td><td class="fml-note">—</td></tr>
        <tr><td class="fml-src std">提取 $s$</td><td class="fml-eq">$= s_{t,e} \cdot (dA'_{e,t} \odot J(H_{e,t}))$</td><td class="fml-note">$dA' = dO \cdot W_2^{\mathsf T}$</td></tr>
        <tr><td class="fml-src sm">SonicMoE</td><td class="fml-eq">$= s_{t,e} \cdot (dA'_{e,t} \odot J(h_{e,t}))$ &nbsp;&nbsp; ✓</td><td class="fml-note"><code>colvec_scale</code> · (dA' ⊙ J(h))</td></tr>
      </table>
      <p class="eq-concl">⇒ $h = H$（是同一个前向缓存），所以两式字面相同。</p>
    </div>

    <p style="margin:14px 0 0;padding:10px 14px;background:#fff5d8;border-left:4px solid #e0b300;color:#5a3f00;font-size:13.5px">
      <b>🎯 小结：</b>两种方案在 $dH$、$dW_1$、$dW_2$、$dS$、$dX$ 五个梯度上<b>逐浮点位相同</b>（FP32 累加，reduction 维度一致）。SonicMoE 只改变了：<br>
      ① <b>计算顺序</b>（用 $\langle dA', A\rangle$ 代替 $\langle dO, Y\rangle$ 来算 $dS$，避免 materialize $Y$）；<br>
      ② <b>scale 注入位置</b>（把 $s$ 因子挪到 $A'$ 和 $dH$ 里，避免 materialize $dY$）；<br>
      ③ <b>$A$ 的来源</b>（反向现场用缓存的 $h$ 重算 SwiGLU，而不是 cache $A$ —— element-wise 零 GEMM FLOPs，与 MoMoE 反向重算 GEMM 是两回事）。<br>
      —— 没有任何近似、没有额外 training FLOP、也没有新的数值不稳定来源。
    </p>
  </div>

  <h4 class="prologue-h4">📐 反向各算子的输入依赖（公式推导谁需要谁）</h4>
  <table class="prologue-tbl">
    <thead><tr><th>反向输出</th><th>公式</th><th>需要的 forward 张量</th><th>Standard MoE</th><th>SonicMoE</th></tr></thead>
    <tbody>
      <tr>
        <td>$dY$</td>
        <td>$dY_{e,t} = s_{t,k} \cdot dO_t$（scatter）</td>
        <td>$s$（永远缓存，小）</td>
        <td>materialize 到 HBM</td>
        <td><b style="color:#1f5d1f">不存在</b>（dS 重排后用不到）</td>
      </tr>
      <tr>
        <td>$dA$</td>
        <td>$dA = dY \cdot W_2^\top$</td>
        <td>$dY, W_2$</td>
        <td>$[TK,I]$ 中间张量</td>
        <td><b style="color:#1f5d1f">直接 = $dA' = dO \cdot W_2^\top$</b>（在 TMEM 内，永不落 HBM）</td>
      </tr>
      <tr>
        <td>$dH$</td>
        <td>$dH = dA \odot J_\text{SwiGLU}(H)$<br><span style="font-size:11px;color:#888">$J_\text{gate}=\sigma(H_g)(1{+}H_g(1{-}\sigma(H_g)))H_u$<br>$J_\text{up}=\mathrm{silu}(H_g)$</span></td>
        <td>$H$（pre-activation）</td>
        <td>cache $H$ ✓</td>
        <td>cache $H$ ✓ <span style="color:#666">(两边一样)</span></td>
      </tr>
      <tr>
        <td>$dW_2$</td>
        <td>$dW_{2,e} = A_e^\top \cdot dY_e$</td>
        <td>$A, dY$</td>
        <td>cache $A$ ⚠ ($[TK,I]$ = 768MB)</td>
        <td><b style="color:#1f5d1f">改写为 $dW_{2,e} = A'^\top \cdot dO_e$</b><br>$A' = \mathrm{SwiGLU}(H)$ 在 dH kernel 内重算（element-wise，免费）</td>
      </tr>
      <tr>
        <td>$dW_1$</td>
        <td>$dW_{1,e} = X_{g,e}^\top \cdot dH_e$</td>
        <td>$X_g$（即 X 按 expert 重排）</td>
        <td>cache $X_g$ ⚠ ($[TK,d]$ = 2GB)</td>
        <td><b style="color:#1f5d1f">cache $X$（$[T,d]$ = 256MB）</b>，varlen-K GEMM 时用 TMA gather4 现场 gather</td>
      </tr>
      <tr>
        <td>$dS$</td>
        <td>$dS_{t,k} = \langle dO_t, Y_{e_k,t}\rangle$</td>
        <td>$dO, Y$</td>
        <td>cache $Y$ ⚠ ($[TK,d]$ = 2GB)</td>
        <td><b style="color:#1f5d1f">改写为 $dS_{t,k} = \langle dA', A\rangle$</b><br>$dA'$ 在 TMEM、$A$ 由 $H$ 重算 ⇒ 不需要 Y！(行内归约 + colvec_reduce)</td>
      </tr>
      <tr>
        <td>$dX$</td>
        <td>$dX_t = \sum_{k} dX_{g, \pi(t,k)}$</td>
        <td>仅需 $dH_e \cdot W_1$</td>
        <td>—</td>
        <td>—</td>
      </tr>
    </tbody>
  </table>

  <p class="prologue-note">
    <b>SonicMoE 的核心算法发明（dS contraction reordering）：</b><br>
    标准做法 $dS = \langle dO, Y\rangle$ 强制把 $Y = A W_2$ materialize 出来。SonicMoE 利用内积结合律：
    $$dS_{t,e} = \langle dO_t, A_{e,t} W_{2,e}\rangle = \langle dO_t W_{2,e}^\top, A_{e,t}\rangle = \langle dA'_{e,t}, A_{e,t}\rangle$$
    $dA'$ 是反向 down-proj GEMM 的天然输出（在 TMEM 里），$A$ 由缓存的 $H$ 现场 SwiGLU 重算（element-wise，零 GEMM FLOP）⇒ <b>$Y$ 与 $dY$ 都不再需要 materialize 到 HBM</b>。这一笔同时干掉了 forward 的 $Y$ cache、forward 的 $A$ cache（标记 <code>mark_non_differentiable</code>）、以及反向的 $dY$ 中间张量 —— 三件事一起。bit-exact，无任何近似。
  </p>

  <!-- =========================================================== -->
  <h4 class="prologue-h4" id="pr-stepwise">📋 Forward / Backward 逐步对照：Standard MoE vs SonicMoE（Qwen3-235B-A22B 实例）</h4>
  <p>全部基于：$T=32768, d=4096, n=I=1536, E=128, K=8$，BF16 激活/权重（2 B/元素），FP32 梯度/optimizer（4 B/元素）。每行列出 <b>kernel 名</b>、<b>HBM 读/写</b>、<b>cache 状态</b>、<b>SonicMoE 做了什么改动</b>。<span style="color:#b85450">红色</span>=$O(TKd)$ cache，<span style="color:#1f5d1f">绿色</span>=SonicMoE 省掉。</p>

  <!-- ============ Forward ============ -->
  <p style="font-weight:700;color:#7a4e00;margin-top:14px;font-size:15px">🔵 Forward Pass —— 5 kernels (Standard) vs 3 kernels (SonicMoE)</p>
  <table class="stepwise-tbl">
    <thead>
      <tr>
        <th style="width:5%">#</th>
        <th style="width:15%">语义算子</th>
        <th style="width:38%">Standard MoE（DeepGEMM+compile 代表）</th>
        <th style="width:38%">SonicMoE</th>
        <th style="width:4%">变化</th>
      </tr>
    </thead>
    <tbody>

    <tr>
      <td>0</td><td>Router linear</td>
      <td>
        <code>F.linear(X, W_r)</code><br>
        READ: X [T,d]=256MB, W_r [d,E]=1MB<br>
        WRITE: logits [T,E]=8MB
      </td>
      <td>同上</td>
      <td>=</td>
    </tr>

    <tr>
      <td>0'</td><td>Top-K 选路</td>
      <td>
        <code>softmax → torch.topk</code>：若干 PyTorch op，先 softmax [T,E] 再 topk<br>
        HBM: ~30 MB
      </td>
      <td>
        <code>Softmax_Over_TopK</code>（CuTeDSL bitonic，<code>topk.py</code>）：1 kernel<br>
        index 编码进 fp32 mantissa 低位 —— values + indices 共享一个 register slot<br>
        HBM: ~16 MB
      </td>
      <td style="color:#1f5d1f">↑</td>
    </tr>

    <tr>
      <td>0''</td><td>Routing metadata</td>
      <td>
        PyTorch 风格 <code>cumsum / argsort / mask</code> 一长串 op，~10 kernels 串行<br>
        HBM: ~20 MB（全是小 read/write）
      </td>
      <td>
        <code>TC_topk_router_metadata_triton</code>：3 段 Triton<br>
        ① tile 直方图 (atomic_add) → ② prefix-sum → ③ sort+scatter<br>
        输出：<code>x_gather_idx</code>, <code>s_scatter_idx</code>, <code>s_reverse_scatter_idx</code>, <code>expert_frequency_offset</code><br>
        HBM: ~5 MB
      </td>
      <td style="color:#1f5d1f">↑</td>
    </tr>

    <tr style="background:#fff5f0">
      <td>1</td><td>Gather tokens by expert</td>
      <td>
        <b>独立 gather kernel</b>（DeepGEMM API 不支持 <code>A_idx</code>）<br>
        READ: X (256 MB) + <code>gather_idx</code><br>
        WRITE: <span style="color:#b85450;font-weight:700">X_g [TK, d] = 2 GB ⚠ cached-for-bwd</span><br>
        HBM: ~2.25 GB
      </td>
      <td>
        <b style="color:#1f5d1f">无独立 kernel</b> —— gather 融进下一步 mainloop<br>
        用 <code>cp.async.bulk.tensor.*.gather4</code> 把 X 的指定行直接搬到 SMEM<br>
        由于 X 只有 256 MB 能 stay in L2，实际 HBM ~200 MB
      </td>
      <td style="color:#1f5d1f">✂</td>
    </tr>

    <tr>
      <td>2</td><td>Up-proj GEMM（varlen-M）</td>
      <td>
        <code>deepgemm.sm100_m_grouped_bf16_gemm(X_g, W1, cu_seqlens)</code><br>
        READ: X_g (2 GB) + W1 [E, 2I, d] = 3 GB<br>
        WRITE: <span style="color:#b85450;font-weight:700">H [TK, 2I] = 1.5 GB ⚠ cached</span><br>
        单 CTA UMMA + 静态 scheduler<br>
        HBM: ~6.5 GB
      </td>
      <td>
        <code>gemm_gated</code>（QuACK，<code>forward.py:82</code>）：<br>
        • producer：TMA gather4(X, A_idx=x_gather_idx) → SMEM（与 mainloop overlap）<br>
        • mainloop：<code>tcgen05.mma cta_group::2</code>，2CTA 共享 B-tile，M_tile=256，累加器入 TMEM<br>
        • epilogue：在 register 内 SwiGLU(gate, up) + <code>st.async</code> 写 h 和 a<br>
        • CLC 动态 scheduler（<code>try_cancel</code>）<br>
        READ: X (~200 MB eff.) + W1 (3 GB) ⇒ HBM ~3.2 GB<br>
        WRITE: <span style="color:#1f5d1f;font-weight:700">h (1.5 GB) ✓cached</span> + <span style="color:#b46504">a (768 MB) NOT saved (<code>mark_non_differentiable</code>)</span>
      </td>
      <td style="color:#1f5d1f">⊕</td>
    </tr>

    <tr style="background:#fff5f0">
      <td>3</td><td>SwiGLU activation</td>
      <td>
        <b>独立 kernel</b>（torch.compile 可能 fuse 但不跨 GEMM 边界）<br>
        READ: H (1.5 GB)<br>
        WRITE: <span style="color:#b85450;font-weight:700">A [TK, I] = 768 MB ⚠ cached</span><br>
        HBM: ~2.27 GB
      </td>
      <td>
        <b style="color:#1f5d1f">已 fuse 在 Step 2 epilogue 里</b> —— 0 HBM 额外
      </td>
      <td style="color:#1f5d1f">✂</td>
    </tr>

    <tr>
      <td>4</td><td>Down-proj GEMM（varlen-M）</td>
      <td>
        <code>deepgemm.sm100_m_grouped_bf16_gemm(A, W2, cu_seqlens)</code><br>
        READ: A (768 MB) + W2 [E, d, I] = 1.5 GB<br>
        WRITE: <span style="color:#b85450;font-weight:700">Y [TK, d] = 2 GB ⚠ cached</span>（反向 dS 需要）<br>
        HBM: ~4.3 GB
      </td>
      <td>
        <code>gemm</code>（QuACK，<code>forward.py:107</code>）：<br>
        • 2CTA UMMA + CLC + <code>st.async.release.global</code><br>
        READ: a (768 MB, 已在 HBM) + W2 (1.5 GB)<br>
        WRITE: <span style="color:#b46504">y [TK, d] = 2 GB ephemeral</span>（立刻被 Step 5 消费，不进 saved-for-bwd）<br>
        HBM: ~4.3 GB（流量同，但不占 activation memory budget）
      </td>
      <td style="color:#1f5d1f">✓</td>
    </tr>

    <tr style="background:#fff5f0">
      <td>5</td><td>Scatter + weighted sum</td>
      <td>
        <b>两步</b>：(a) scatter Y → Y_scattered；(b) weighted sum 到 O<br>
        READ: Y (2 GB) + <code>scatter_idx</code> + <code>topk_scores</code><br>
        WRITE: <span style="color:#b85450;font-weight:700">Y_scattered [TK, d] = 2 GB ⚠ cached</span> + O [T, d] = 256 MB<br>
        HBM: ~4.5 GB（scatter 用 atomic 时反向 kernel 还会再读 scatter_idx）
      </td>
      <td>
        <code>token_gather_and_sum_varlen_K_triton</code>（<code>reduction_over_k_gather.py</code>）：<br>
        每 token gather 自己的 K 个 y 片段再 weighted-sum：<br>
        $O_t = \sum_{k=1}^{K} s_{t,k} \cdot y[\text{rev\_scat\_idx}[t\cdot K + k]]$<br>
        READ: y (2 GB) + topk_scores + rev_scat_idx<br>
        WRITE: O [T, d] = 256 MB<br>
        <b>6.5+ TB/s（&gt;85% 峰值 HBM 带宽）</b><br>
        HBM: ~2.3 GB（<span style="color:#1f5d1f">没有 Y_scattered materialize</span>）
      </td>
      <td style="color:#1f5d1f">✂</td>
    </tr>

    <tr style="background:#fff8c4;font-weight:700">
      <td colspan="2">Forward 合计</td>
      <td>
        5 kernels（+ 独立 gather 算 6）<br>
        HBM ~17 GB / 层<br>
        <span style="color:#b85450">Cache: X_g (2GB) + H (1.5GB) + A (768MB) + Y (2GB) + Y_scat (2GB) ≈ 8.3 GB / 层</span>
      </td>
      <td>
        <b>3 kernels</b>（gemm_gated + gemm + token_gather_sum）<br>
        HBM ~7 GB / 层<br>
        <span style="color:#1f5d1f">Cache: X (256MB) + h (1.5GB) + 路由 metadata (5MB) ≈ 1.8 GB / 层</span>
      </td>
      <td></td>
    </tr>

    </tbody>
  </table>

  <!-- ============ Backward activation gradient ============ -->
  <p style="font-weight:700;color:#7a4e00;margin-top:20px;font-size:15px">🔴 Backward — Activation Gradient Path（$dO \to dX$）</p>
  <table class="stepwise-tbl">
    <thead>
      <tr>
        <th style="width:5%">#</th>
        <th style="width:15%">语义算子</th>
        <th style="width:38%">Standard MoE</th>
        <th style="width:38%">SonicMoE</th>
        <th style="width:4%">变化</th>
      </tr>
    </thead>
    <tbody>

    <tr style="background:#fff5f0">
      <td>B1</td><td>Gather dO</td>
      <td>
        <b>独立 gather kernel</b><br>
        READ: dO [T, d] = 256 MB + <code>gather_idx</code><br>
        WRITE: <span style="color:#b85450">dO_g [TK, d] = 2 GB</span>（临时）<br>
        HBM: ~2.25 GB
      </td>
      <td>
        <b style="color:#1f5d1f">fused 进 dH kernel 的 producer warp</b>（同一份 TMA gather4）
      </td>
      <td style="color:#1f5d1f">✂</td>
    </tr>

    <tr style="background:#fff5f0">
      <td>B2</td><td>compute dY</td>
      <td>
        $dY_{e,t} = s_{t,k} \cdot dO_t$（scatter from dO by routing）<br>
        WRITE: <span style="color:#b85450;font-weight:700">dY [TK, d] = 2 GB</span> materialize<br>
        HBM: ~2.25 GB
      </td>
      <td>
        <b style="color:#1f5d1f">dY 不存在</b> —— 下一步 B3 用 dA' 代替，不需要显式 dY
      </td>
      <td style="color:#1f5d1f">✂</td>
    </tr>

    <tr>
      <td>B3</td><td>dA = dY @ W₂ᵀ<br>（或 dA' = dO @ W₂ᵀ）</td>
      <td>
        varlen-M grouped GEMM<br>
        READ: dY (2 GB) + W2 (1.5 GB)<br>
        WRITE: dA [TK, I] = 768 MB<br>
        HBM: ~4.3 GB
      </td>
      <td>
        <b>dH kernel mainloop</b>（<code>gemm_dgated</code>）：<br>
        • Producer：TMA gather4(dO, A_idx=x_gather_idx)<br>
        • MMA：tcgen05.mma → dA' = dO·W₂ᵀ <b style="color:#1f5d1f">写 TMEM，永不落 HBM</b><br>
        READ: dO (~200 MB eff.) + W2 (1.5 GB)<br>
        HBM: ~1.7 GB
      </td>
      <td style="color:#1f5d1f">⊕</td>
    </tr>

    <tr>
      <td>B4</td><td>dH = dA ⊙ dSwiGLU(H)</td>
      <td>
        Element-wise kernel<br>
        READ: dA (768 MB) + <span style="color:#b85450">H (1.5 GB, cached from fwd)</span><br>
        WRITE: dH [TK, 2I] = 1.5 GB<br>
        HBM: ~3.8 GB
      </td>
      <td>
        <b>dH kernel epilogue</b>，在 register 内完成：<br>
        • TMA-load h-tile（≤ SMEM, tiled）<br>
        • 重算 $A = \mathrm{SwiGLU}(h)$ （element-wise）<br>
        • 计算 dSwiGLU jacobian → $dH = dA' \odot J$<br>
        • <code>st.async.release.global</code> 写 dH<br>
        READ: h (1.5 GB)<br>
        WRITE: <span style="color:#1f5d1f">dH (1.5 GB)</span>
      </td>
      <td style="color:#1f5d1f">⊕</td>
    </tr>

    <tr style="background:#fff5f0">
      <td>B5</td><td>dS = ⟨dO, Y⟩</td>
      <td>
        行内点积，需要 <span style="color:#b85450;font-weight:700">cached Y</span><br>
        READ: dO + Y (2 GB)<br>
        WRITE: dS [T, K] (小)<br>
        HBM: ~2.3 GB
      </td>
      <td>
        <b style="color:#1f5d1f">fused 进 dH epilogue 的 <code>colvec_reduce</code></b>：<br>
        $dS_{\text{scattered}} = \text{rowsum}(dA' \odot A) \cdot s$<br>
        → <code>ds[s_scatter_idx] = ds_scattered</code><br>
        0 HBM 额外（都是 dH epilogue 里已读的张量）
      </td>
      <td style="color:#1f5d1f">✂</td>
    </tr>

    <tr>
      <td>B6</td><td>A' = SwiGLU(h) 重算<br>（用于 dW2）</td>
      <td>不需要重算 —— 直接用 cached A</td>
      <td>
        <b>同样在 dH epilogue</b>：<code>postact_out=a_prime</code> 把重算的 $A$ 写到 HBM，喂给 dW2 kernel<br>
        WRITE: a_prime (768 MB)
      </td>
      <td style="color:#b46504">+</td>
    </tr>

    <tr>
      <td>B7</td><td>dX 聚合</td>
      <td>
        grouped GEMM: dX_g = dH @ W1<br>
        + scatter_sum over K<br>
        READ: dH + W1 + scatter_idx<br>
        WRITE: dX [T, d] = 256 MB
      </td>
      <td>
        同：<code>_up_projection_backward_act</code> 用 <code>gemm</code> + <code>_token_broadcast_backward</code> Triton 做 reverse scatter+sum
      </td>
      <td>=</td>
    </tr>

    <tr style="background:#fff8c4;font-weight:700">
      <td colspan="2">Backward-act 合计</td>
      <td>
        <b>5-6 kernels</b>（gather dO + scatter dY + dA GEMM + dH + dS + dX sum）<br>
        HBM ~15 GB / 层
      </td>
      <td>
        <b>2 kernels</b>（<code>gemm_dgated</code> 一发出 dH+A'+dS + reverse-scatter Triton 算 dX）<br>
        HBM ~7.86 GB（NCU 实测）<br>
        <span style="color:#1f5d1f">无 $dY$，无 $Y$ 读，无额外 GEMM recompute</span>
      </td>
      <td></td>
    </tr>

    </tbody>
  </table>

  <!-- ============ Backward weight gradient ============ -->
  <p style="font-weight:700;color:#7a4e00;margin-top:20px;font-size:15px">🟣 Backward — Weight Gradient Path（$dW_1, dW_2$）</p>
  <table class="stepwise-tbl">
    <thead>
      <tr>
        <th style="width:5%">#</th>
        <th style="width:15%">语义算子</th>
        <th style="width:38%">Standard MoE</th>
        <th style="width:38%">SonicMoE</th>
        <th style="width:4%">变化</th>
      </tr>
    </thead>
    <tbody>

    <tr>
      <td>W1</td><td>dW₂ = Aᵀ · dY</td>
      <td>
        varlen-K grouped GEMM<br>
        READ: <span style="color:#b85450">A (768 MB, cached)</span> + dY (2 GB)<br>
        WRITE: dW₂ [E, d, I] = 3 GB FP32<br>
        HBM: ~5.8 GB
      </td>
      <td>
        <code>gemm</code> varlen-K（<code>backward.py:325</code>）改写为 $dW_2 = dO^\top \cdot A'$：<br>
        READ: dO (256 MB) + a_prime (768 MB, 来自 dH kernel) + W2 layout<br>
        + <code>cu_seqlens_k=expert_frequency_offset</code> + <code>A_idx=x_gather_idx</code> 内联 gather dO<br>
        WRITE: dW₂ (3 GB FP32)<br>
        HBM: ~4 GB
      </td>
      <td style="color:#1f5d1f">⊕</td>
    </tr>

    <tr>
      <td>W2</td><td>dW₁ = X_gᵀ · dH</td>
      <td>
        varlen-K grouped GEMM<br>
        READ: <span style="color:#b85450">X_g (2 GB, cached)</span> + dH (1.5 GB)<br>
        WRITE: dW₁ [E, 2I, d] = 6 GB FP32<br>
        HBM: ~9.5 GB
      </td>
      <td>
        <code>gemm</code> varlen-K（<code>backward.py:225</code>）：<br>
        READ: X (256 MB, 内联 TMA gather4) + dH (1.5 GB)<br>
        WRITE: dW₁ (6 GB FP32)<br>
        HBM: ~7.7 GB
      </td>
      <td style="color:#1f5d1f">⊕</td>
    </tr>

    <tr style="background:#fff8c4;font-weight:700">
      <td colspan="2">Backward-weight 合计</td>
      <td>
        2 varlen-K GEMM<br>
        HBM ~15 GB / 层（其中 X_g+A 从 activation cache 读）
      </td>
      <td>
        2 varlen-K GEMM + 2 次 TMA gather4（内联）<br>
        HBM ~11.7 GB / 层<br>
        <span style="color:#1f5d1f">X / dO 直接从 compact 源读，L2 友好</span>
      </td>
      <td></td>
    </tr>

    </tbody>
  </table>

  <!-- ============ Summary ============ -->
  <p style="font-weight:700;color:#7a4e00;margin-top:20px;font-size:15px">📊 三路汇总（Qwen3-235B-A22B 单层，microbatch=32k）</p>
  <table class="stepwise-tbl">
    <thead>
      <tr>
        <th>指标</th>
        <th style="width:26%">Standard MoE</th>
        <th style="width:26%">SonicMoE</th>
        <th style="width:20%">差值</th>
        <th style="width:16%">博客实测</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Fwd kernel 启动数</td><td>5–6</td><td>3</td><td>−40~50%</td><td>—</td></tr>
      <tr><td>Fwd HBM 流量</td><td>~17 GB</td><td>~7 GB</td><td>−59%</td><td>Fwd TFLOPS +54% vs DeepGEMM</td></tr>
      <tr><td>Bwd-act kernel 启动数</td><td>5–6</td><td>2 (<code>gemm_dgated</code> + reverse-scatter)</td><td>−60%</td><td>—</td></tr>
      <tr><td>Bwd-act HBM 流量</td><td>~15 GB</td><td>~7.86 GB（NCU 实测）</td><td>−48%</td><td>Bwd TFLOPS +35% vs DeepGEMM</td></tr>
      <tr><td>Bwd-weight HBM 流量</td><td>~15 GB</td><td>~11.7 GB</td><td>−22%</td><td>M 维 gather 仅 −1.4%；K 维 +0.5%</td></tr>
      <tr style="background:#fff8c4;font-weight:700">
        <td>Cache activation / 层</td>
        <td>X_g + H + A + Y + Y_scat ≈ <b>8.3 GB</b></td>
        <td>X + h ≈ <b>1.8 GB</b></td>
        <td style="color:#1f5d1f">−78%</td>
        <td>—</td>
      </tr>
      <tr style="background:#fff8c4;font-weight:700">
        <td>× 94 层 × ZeRO 未切分时</td>
        <td>~780 GB</td>
        <td>~170 GB</td>
        <td style="color:#1f5d1f">−610 GB</td>
        <td>可让 micro-batch 或 K 翻倍</td>
      </tr>
      <tr>
        <td>Tensor Core util（dH kernel）</td>
        <td>~75-80%（epilogue 阻塞）</td>
        <td>88%（MMA 与 epilogue IO overlap）</td>
        <td>+10 p.p.</td>
        <td>NCU: TMEM util 也是 88%</td>
      </tr>
      <tr>
        <td>dS 算法</td>
        <td>$\langle dO, Y\rangle$，必须 cache Y</td>
        <td>$\langle dA', A\rangle$，A 现场 SwiGLU 重算</td>
        <td>质变</td>
        <td>bit-exact 等价</td>
      </tr>
    </tbody>
  </table>

  <p class="prologue-note">
    <b>如何读这三张表：</b>
    每一行语义算子（Step 0–5 forward / B1–B7 bwd-act / W1–W2 bwd-weight）在 Standard 侧都对应一个独立的 kernel 边界；SonicMoE 把它们要么<b style="color:#1f5d1f">✂消灭</b>（算法层重排）、要么<b style="color:#1f5d1f">⊕融合</b>进隔壁 GEMM 的 producer/epilogue（软件抽象 + 硬件异步）、要么<b style="color:#b46504">+新增一个重算步骤</b>（$A'=\mathrm{SwiGLU}(h)$，element-wise 几乎免费）。
    <br><br>
    最终观感：Standard 的反向 activation gradient 是 "B1→B2→B3→B4→B5→B7" 六件事六个 kernel，SonicMoE 压成一个 <code>gemm_dgated</code>（源码：<code>sonicmoe/functional/backward.py:262-275</code>）。这就是 SonicMoE 论文里"dH kernel 同时输出 dH/A'/dS"那张图的精确展开。
  </p>

  <!-- ========================================= -->
  <h3 class="prologue-h3" id="pr-s2">② NVIDIA GPU 执行层级（Hopper/Blackwell 通用）</h3>
  <table class="prologue-tbl">
    <thead><tr><th>层级</th><th>含义</th><th>典型规模</th><th>SonicMoE 里的角色</th></tr></thead>
    <tbody>
      <tr><td><b>Grid</b></td><td>整个 kernel 启动的所有 CTA 集合</td><td>数千个 CTA</td><td>由 tile scheduler 分配 tile</td></tr>
      <tr><td><b>Cluster</b></td><td>在同一 GPC 内可共享 SMEM 的一组 CTA（SM90 引入，SM100 扩展）</td><td>通常 size = 1 或 2</td><td>2CTA MMA 需要 cluster size = 2</td></tr>
      <tr><td><b>CTA</b> (Thread Block)</td><td>运行在单个 SM 上的一组 thread，独占 SMEM</td><td>128–512 threads</td><td>通常 1 CTA / SM，跑一个 tile 的完整 prologue+mainloop+epilogue</td></tr>
      <tr><td><b>Warpgroup</b></td><td>4 个连续 warp = 128 threads（Hopper WGMMA 的执行单位）</td><td>128 threads</td><td>Hopper Ping-Pong 2 个 WG 互换 MMA / epilogue 角色</td></tr>
      <tr><td><b>Warp</b></td><td>SIMT 执行单位</td><td>32 threads</td><td>Producer / MMA / Epilogue / Relay / Scheduler warp 各司其职</td></tr>
      <tr><td><b>Thread</b></td><td>单个执行流</td><td>1</td><td>Blackwell UMMA 只需 1 个 thread issue</td></tr>
    </tbody>
  </table>

  <!-- ========================================= -->
  <h3 class="prologue-h3" id="pr-s3">③ 内存层级与带宽（以 B300 为基准）</h3>
  <table class="prologue-tbl">
    <thead><tr><th>层级</th><th>容量</th><th>带宽</th><th>谁能访问</th><th>SonicMoE 用法</th></tr></thead>
    <tbody>
      <tr><td><b>Register</b></td><td>~64K × 32-bit / SM</td><td>~100+ TB/s</td><td>单 thread 专有</td><td>Hopper WGMMA 累加器、所有 epilogue 运算</td></tr>
      <tr><td><b>SMEM</b> (Shared Memory)</td><td>228 KB / SM</td><td>~30 TB/s</td><td>单 CTA 内所有 thread 共享；cluster 内可 multicast</td><td>A-buffer / B-buffer 多 stage 流水；cluster 内 TMA multicast 共享 B-tile</td></tr>
      <tr><td><b>TMEM</b> (Tensor Memory)</td><td>256 KB / SM <span style="color:#d6336c">(Blackwell 新增)</span></td><td>接到 Tensor Core 直连</td><td>Tensor Core + <code>tcgen05.ld/st</code></td><td>UMMA 累加器双 buffer（stage 0 / stage 1）</td></tr>
      <tr><td><b>L2 Cache</b></td><td>192 MB / GPU</td><td>~20 TB/s</td><td>所有 SM 共享</td><td>Gather fusion 通过 L2 命中率差异省 HBM 流量（见 §4）</td></tr>
      <tr><td><b>HBM</b> (Device DRAM)</td><td>288 GB / GPU (HBM3e)</td><td><b>7.7 TB/s</b></td><td>所有 SM + Host 通过 PCIe/NVLink</td><td>最慢的那一级 —— kernel runtime 几乎全由 HBM 流量决定</td></tr>
      <tr><td><b>NVLink</b></td><td>—</td><td>~0.9 TB/s</td><td>同节点 GPU 间</td><td>EP / TP 通信路径（比 HBM 慢 8×）</td></tr>
      <tr><td><b>IB / RoCE</b></td><td>—</td><td>~0.4 TB/s</td><td>跨节点</td><td>EP 跨节点 all-to-all（比 HBM 慢 19×）</td></tr>
    </tbody>
  </table>
  <p class="prologue-note"><b>关键阈值：</b>B300 算力 ≈ 2.5 PFLOPs BF16，HBM 7.7 TB/s ⇒ 算术强度分水岭 $\approx 325$ FLOP/byte。Qwen3-Next-80B-A3B 的 MoE expert 在 16K microbatch 下 AI ≈ 210 &lt; 325 ⇒ <b>memory-bound，优化就是"少读少写 HBM"</b>。</p>

  <!-- ========================================= -->
  <h3 class="prologue-h3" id="pr-s4">④ Tensor Core 指令家族演进</h3>
  <table class="prologue-tbl">
    <thead><tr><th>世代</th><th>指令</th><th>Issue 单位</th><th>累加器位置</th><th>异步性</th><th>CTA 协同</th></tr></thead>
    <tbody>
      <tr><td>Ampere (SM80)</td><td><code>mma.sync.aligned</code></td><td>1 warp</td><td>Register</td><td>同步</td><td>单 CTA</td></tr>
      <tr><td>Hopper (SM90)</td><td><code>wgmma.mma_async</code> <b>WGMMA</b></td><td>1 warpgroup (128 threads)</td><td>Register（分布 128 线程）</td><td>Async，用 fence 同步</td><td>单 CTA</td></tr>
      <tr><td>Blackwell (SM100)</td><td><code>tcgen05.mma</code> <b>UMMA</b></td><td>1 thread</td><td><b>TMEM</b> 256 KB</td><td>Async，用 accumulator pipeline 同步</td><td><b>支持 <code>cta_group::2</code></b>（2CTA cooperative）</td></tr>
    </tbody>
  </table>

  <h4 class="prologue-h4">数据搬运指令</h4>
  <table class="prologue-tbl">
    <thead><tr><th>指令</th><th>方向</th><th>完成事件可见范围</th><th>用途</th></tr></thead>
    <tbody>
      <tr><td><code>cp.async.ca/cg.shared.global</code> (SM80)</td><td>GMEM → SMEM</td><td>CTA-local (<code>commit_group/wait_group</code>)</td><td>fine-grained load，但需要手动搭桥到 cluster</td></tr>
      <tr><td><code>cp.async.bulk.tensor.tile</code> (SM90 TMA)</td><td>GMEM → SMEM / SMEM → GMEM</td><td>Cluster-scope (mbarrier)</td><td>块加载 / 块 store（contiguous tile）</td></tr>
      <tr><td><code>cp.async.bulk.tensor.*.gather4</code> <span style="color:#d6336c">(SM100)</span></td><td>GMEM → SMEM</td><td>Cluster-scope</td><td>一条指令 gather 4 行任意 index（SonicMoE gather fusion 核心）</td></tr>
      <tr><td><code>tcgen05.ld / tcgen05.st</code> <span style="color:#d6336c">(SM100)</span></td><td>TMEM ↔ Register</td><td>Async</td><td>epilogue drain 累加器</td></tr>
      <tr><td><code>st.async.release.global</code> <span style="color:#d6336c">(SM100)</span></td><td>Register → GMEM</td><td>Async</td><td>epilogue store 不阻塞下一 tile 的 MMA</td></tr>
      <tr><td><code>clusterlaunchcontrol.try_cancel</code> <span style="color:#d6336c">(SM100)</span></td><td>—</td><td>Cluster-scope</td><td>CLC 动态 tile 调度，无 GMEM atomics</td></tr>
    </tbody>
  </table>

  <!-- ========================================= -->
  <h3 class="prologue-h3" id="pr-s5">⑤ 本文涉及的 Hopper / Blackwell 优化点全景（SonicMoE 视角）</h3>
  <p>§3 与 §4 会详细介绍每项。本表提前给出全景，方便在读正文时"按图索骥"。所有"量化收益"列都来自论文正文或附录实测数字。</p>

  <table class="prologue-tbl">
    <thead>
    <tr>
      <th style="width:14%">优化点</th>
      <th style="width:22%"><span style="color:#2b8a3e">Hopper (SM90)</span> 做法</th>
      <th style="width:24%"><span style="color:#d6336c">Blackwell (SM100)</span> 做法</th>
      <th style="width:22%">SonicMoE 使用点</th>
      <th style="width:18%">量化收益 / 依据</th>
    </tr>
    </thead>
    <tbody>

    <tr>
      <td><b>① MMA 指令</b></td>
      <td><code>wgmma.mma_async</code><br>warpgroup (128 threads) 一起 issue</td>
      <td><code>tcgen05.mma</code> (UMMA)<br><b>单 thread async issue</b>，不占用其他线程</td>
      <td>两种架构各自 base class 里切换；epilogue Mixin 代码完全共享</td>
      <td>UMMA 释放 WG 其他线程做 producer / scheduler —— warp specialization 基础</td>
    </tr>

    <tr>
      <td><b>② 累加器位置</b></td>
      <td>分布在 128 线程的 <b>register</b> 中</td>
      <td><b>TMEM</b> (256 KB / SM)，两个 256-列 stage 天然双 buffer</td>
      <td>dH kernel 利用 TMEM stage 0/1 交替；epilogue 读一侧时 MMA 写另一侧</td>
      <td>dH: HBM +24% 但 TFLOPS 仅 −11%（亚比例下降，§4）</td>
    </tr>

    <tr>
      <td><b>③ IO / MMA overlap</b></td>
      <td><b>Ping-Pong warpgroup</b>：2 个 WG 互换 MMA 与 epilogue 角色，交替跑</td>
      <td><b>TMEM 双 buffer</b>：1 MMA warp + 多个 epilogue warp 并发，stage 在 tile 间交替</td>
      <td>Hopper 走 Ping-Pong，Blackwell 走 warp-specialized pipeline；QuACK <code>epi_visit_subtile</code> 两者共用</td>
      <td>Blackwell 省掉 WG-级别的 register 压力翻倍</td>
    </tr>

    <tr>
      <td><b>④ CTA 协同</b></td>
      <td>单 CTA MMA（即便有 cluster，每个 CTA 仍然各自累加）</td>
      <td><b>2CTA UMMA</b> (<code>cta_group::2</code>)：一条 MMA 指令跨 2 个 CTA；<b>B-tile 通过 TMA multicast 共享</b>，每 CTA 只 load 一半 B</td>
      <td>varlen-M Grouped GEMM 默认开 2CTA；varlen-K 按 shape autotune 决定</td>
      <td>B 侧 SMEM traffic 减半 ⇒ 算术强度提升 ≈ 2×；贡献 +54% (vs DeepGEMM) 中的 ~7-10%</td>
    </tr>

    <tr>
      <td><b>⑤ Tile 调度</b></td>
      <td>静态 linear 预分配（零同步但 MoE 长尾不均）或软件 GMEM atomic queue（开销大）</td>
      <td><b>CLC</b> <code>clusterlaunchcontrol.try_cancel</code>：硬件辅助 cluster-level 动态 tile 调度，无 GMEM atomics，响应广播给整 cluster</td>
      <td><code>SonicMoEVarlenMTileScheduler</code> 扩展 QuACK base scheduler 加 prefetch（<code>sonicmoe/functional/tile_scheduler.py</code>）</td>
      <td>消灭 MoE 长 expert 的 tail latency；plain Grouped GEMM 贡献 ~3-5% 吞吐提升</td>
    </tr>

    <tr>
      <td><b>⑥ Gather fusion</b></td>
      <td><code>cp.async</code>（CTA-local 完成事件，需要 relay warp 桥接到 cluster barrier）</td>
      <td><b>TMA gather4</b> <code>cp.async.bulk.tensor.*.gather4</code>：一条指令搬 4 行，完成事件挂在 cluster-scope mbarrier 上</td>
      <td>gather 路径（cp.async vs TMA gather4）作为 <b>autotunable config</b>（实测 &lt; 2% 差异）；2CTA + cp.async 时走 relay warp</td>
      <td>M 维 gather fusion 仅慢 1.4%、K 维反而快 0.5% vs contiguous；+25-30% (vs DeepGEMM) 主要来源</td>
    </tr>

    <tr>
      <td><b>⑦ L2 Cache locality</b></td>
      <td>L2 60 MB</td>
      <td>L2 192 MB（仍可能被预 gather 的 $X_g$ 撑爆）</td>
      <td><b>不预 gather</b> $X_g$ —— 从原始 $X$ 内联 gather，source tensor 小 K 倍 ⇒ 更可能 stay in L2</td>
      <td>up-proj fwd 实测：HBM 2.20 vs 2.68 GB；L2 hit 74.9% vs 66.3% (appendix)</td>
    </tr>

    <tr>
      <td><b>⑧ Epilogue Store</b></td>
      <td>同步 <code>st.global</code> / 阻塞 TMA store —— scatter fusion 在 fine-grained MoE 上让 TFLOPS 降 20%</td>
      <td><b><code>st.async.release.global</code></b> 与 <b>TMA scatter4</b>：async store 不阻塞 accumulator pipeline</td>
      <td>dH kernel epilogue 的三路 store (dH / A' / dS) 都走 async，不拖累下一 tile 的 MMA</td>
      <td>GEMM w. scatter 与 GEMM + gather-and-sum 的差距从 Hopper 20% 收窄到 Blackwell 3%</td>
    </tr>

    <tr>
      <td><b>⑨ Warp specialization</b></td>
      <td>1 producer WG + 2 consumer WGs（Ping-Pong）</td>
      <td><b>多角色并发</b>：1 producer + 1 MMA + N epilogue + 1 scheduler warp；可以再让 1 个 warp 专门给 epilogue 做 TMA-load</td>
      <td>dH kernel 在 epilogue 内部嵌套了"epilogue 内部的 producer-consumer"—— 一个 warp 专门 TMA-load $H$</td>
      <td>支撑 dH kernel 把 4 个 epilogue ops 装进去却只掉 11% TFLOPS</td>
    </tr>

    <tr>
      <td><b>⑩ SMEM Multicast</b></td>
      <td>Cluster TMA multicast 存在但 WGMMA 不能 cross-CTA 协同 ⇒ 用不起来</td>
      <td>TMA multicast <b>+</b> 2CTA UMMA ⇒ B-tile 真正在 cluster 内共享 SMEM traffic</td>
      <td>见 ④</td>
      <td>见 ④</td>
    </tr>

    <tr>
      <td><b>⑪ cp.async 完成事件</b></td>
      <td>CTA-local（<code>commit_group / wait_group</code>）</td>
      <td>同 Hopper 的 cp.async（TMA 才有 cluster-scope mbarrier）</td>
      <td>cp.async + 2CTA 必须引入 <b>relay warp</b> 把 CTA-local 完成事件 forward 到 cluster barrier（图见 §4）</td>
      <td>踩坑经验：relay 不能复用 producer（会 deadlock），必须独立 1 warp</td>
    </tr>

    </tbody>
  </table>

  <!-- Three-layer diagram -->
  <h4 class="prologue-h4">🎯 三层优化如何叠加成最终 +54% / +35%</h4>
  <div class="svg-wrapper">
  <svg viewBox="0 0 980 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#fffcf1;border:1px solid #d9b860;border-radius:4px;">
    <!-- Algorithm layer -->
    <rect x="30" y="20" width="920" height="60" rx="6" fill="#f8cecc" stroke="#b85450"/>
    <text x="50" y="42" font-family="sans-serif" font-size="13" font-weight="700" fill="#721c24">算法层 (Algorithm)</text>
    <text x="50" y="62" font-family="sans-serif" font-size="12" fill="#721c24">消灭所有 O(TKd) cache · dS contraction reorder · A inline 重算 · forward 3 kernel + backward 3 kernel</text>
    <text x="940" y="62" font-family="sans-serif" font-size="11" text-anchor="end" fill="#721c24" font-weight="700">activation −85%</text>

    <!-- Software layer -->
    <rect x="30" y="95" width="920" height="60" rx="6" fill="#fff2cc" stroke="#d6b656"/>
    <text x="50" y="117" font-family="sans-serif" font-size="13" font-weight="700" fill="#5a3f00">软件层 (QuACK)</text>
    <text x="50" y="137" font-family="sans-serif" font-size="12" fill="#5a3f00">三段式流水 · epi_visit_subtile 单注入点 · 算术 Mixin × 架构 Base · SonicMoE 仅 200 LoC 跨 SM90/100</text>
    <text x="940" y="137" font-family="sans-serif" font-size="11" text-anchor="end" fill="#5a3f00" font-weight="700">跨架构代价 &lt; 100 LoC / 特性</text>

    <!-- Hardware layer -->
    <rect x="30" y="170" width="920" height="60" rx="6" fill="#d5e8d4" stroke="#5fa55f"/>
    <text x="50" y="192" font-family="sans-serif" font-size="13" font-weight="700" fill="#1f5d1f">硬件层 (Blackwell)</text>
    <text x="50" y="212" font-family="sans-serif" font-size="12" fill="#1f5d1f">UMMA + TMEM 双 buffer · 2CTA shared-B · CLC 调度 · TMA gather4 · async store · cluster barrier</text>
    <text x="940" y="212" font-family="sans-serif" font-size="11" text-anchor="end" fill="#1f5d1f" font-weight="700">fwd +54% · bwd +35%</text>

    <!-- Arrows showing stacking -->
    <g stroke="#888" stroke-width="1.5" fill="none" stroke-dasharray="3,3">
      <line x1="490" y1="80" x2="490" y2="95" marker-end="url(#arr-dep)"/>
      <line x1="490" y1="155" x2="490" y2="170" marker-end="url(#arr-dep)"/>
    </g>
  </svg>
  </div>
  <p class="prologue-note" style="margin-top:8px">
    <b>为什么"算法 + 软件 + 硬件" 这种堆叠能跨架构复用？</b> 因为 <b>QuACK 的 <code>epi_visit_subtile</code> 单注入点</b>把"MoE-specific 算什么"与"底层硬件用哪条 MMA 指令、累加器放哪儿、怎么调度"彻底解耦 —— SM90 / SM100 / SM120 各自的 Base 类只负责第二块，epilogue 里的算术逻辑一份代码跑所有架构。从 Hopper 移植到 Blackwell 的 TMA gather4 仅 ~100 LoC、SM120 扩展仅 ~500 LoC，靠的就是这条缝画得对。
  </p>

  <!-- ========================================= -->
  <h3 class="prologue-h3" id="pr-s6">⑥ Grouped GEMM / varlen-M / varlen-K</h3>
  <p>一批形状可能不同的矩阵乘。沿用 CUTLASS 的 BLAS 约定 $C_e = A_e B_e$，$A_e \in \mathbb{R}^{M_e \times K_e}$、$B_e \in \mathbb{R}^{K_e \times N_e}$、$C_e \in \mathbb{R}^{M_e \times N_e}$。</p>
  <ul>
    <li><b>varlen-M Grouped GEMM</b>：$N, K$ 固定，$M_e$ 随 expert 变化。对应 MoE 的 <b>forward up-proj / down-proj</b>、<b>backward activation gradient (dH)</b>。用 <code>cu_seqlens_m</code>（exclusive prefix-sum）传边界。</li>
    <li><b>varlen-K Grouped GEMM</b>：$M, N$ 固定（embedding dim 与 intermediate dim），$K_e$ 随 expert 变化。对应 MoE 的 <b>backward weight gradient (dW1, dW2)</b>。split-K 不适用，用 persistent kernel + per-expert prologue。</li>
  </ul>

  <!-- ========================================= -->
  <h3 class="prologue-h3" id="pr-s7">⑦ 软件栈：CUTLASS / CuTeDSL / QuACK</h3>
  <ul>
    <li><b>CUTLASS</b>：NVIDIA 的 C++ 模板库，把 GEMM 拆成 <b>tile → thread → warp → warpgroup</b> 分层的模板。</li>
    <li><b>CuTeDSL</b>：CUTLASS 的 DSL（Python + JIT compile 到 PTX），统一 GMEM/SMEM/TMEM/Register 之间 copy 的 atom 抽象。<b>"换硬件只换 atom"</b>的关键。</li>
    <li><b>QuACK</b>：SonicMoE 作者团队基于 CuTeDSL 的自研库（<code>quack/</code> 子模块），在上面加了 tile scheduler、customizable epilogue 等模块。SonicMoE 的 <code>gemm / gemm_gated / gemm_dgated</code> API 都来自 QuACK。</li>
    <li><b>QuACK 的三段式</b>：Prologue (producer 加载 SMEM) → Mainloop (MMA 累加) → Epilogue (<code>epi_visit_subtile</code> 注入 fusion + 写 GMEM)。所有 SonicMoE kernel 的 MoE-specific 逻辑都只在 <code>epi_visit_subtile</code> 里。</li>
  </ul>

  <!-- ========================================= -->
  <h3 class="prologue-h3" id="pr-s8">⑧ 符号速查表</h3>
  <table class="prologue-tbl">
    <thead><tr><th>符号</th><th>含义</th><th>示例值 (Qwen3-235B-A22B)</th></tr></thead>
    <tbody>
      <tr><td>$T$</td><td>microbatch 内 token 总数</td><td>32 768</td></tr>
      <tr><td>$d$</td><td>embedding (hidden) dimension</td><td>4 096</td></tr>
      <tr><td>$n$ 或 $I$</td><td>单 expert 的 intermediate dimension</td><td>1 536</td></tr>
      <tr><td>$E$</td><td>expert 总数</td><td>128</td></tr>
      <tr><td>$K$</td><td>每 token 激活 expert 数 (top-K)</td><td>8</td></tr>
      <tr><td>$TK$</td><td>grouped token 总数（每 token 复制 K 次）</td><td>262 144</td></tr>
      <tr><td>$G = d/n$</td><td><b>Expert Granularity</b>（越大越 fine-grained）</td><td>2.67</td></tr>
      <tr><td>$\rho = K/E$</td><td><b>Sparsity</b>（越小越稀疏）</td><td>0.0625</td></tr>
      <tr><td>$M_e$</td><td>expert $e$ 收到的 token 数 (varlen)</td><td>平均 $T\rho$ ≈ 2048</td></tr>
      <tr><td colspan="3" style="background:#fafafa;padding:6px 10px;"><b>Forward 张量</b></td></tr>
      <tr><td>$X$</td><td>MoE 层输入 activation</td><td>$[T, d]$ BF16 = 256 MB</td></tr>
      <tr><td>$X_g$</td><td>gathered 输入（按 expert 分组）</td><td>$[TK, d]$ = 2 GB ⚠ SonicMoE 不 materialize</td></tr>
      <tr><td>$H$</td><td>up-proj 输出（pre-activation）</td><td>$[TK, 2I]$ BF16 = 1.5 GB ✓ SonicMoE 唯一缓存</td></tr>
      <tr><td>$A$</td><td>post-activation（SwiGLU(H)）</td><td>$[TK, I]$ BF16 = 768 MB，non-differentiable</td></tr>
      <tr><td>$Y$</td><td>down-proj 输出</td><td>$[TK, d]$ = 2 GB ⚠ SonicMoE 不缓存（dS 重排）</td></tr>
      <tr><td>$s$</td><td>路由 score (top-K probs)</td><td>$[T, K]$ FP32</td></tr>
      <tr><td>$O$</td><td>MoE 层最终输出</td><td>$[T, d]$ BF16</td></tr>
      <tr><td colspan="3" style="background:#fafafa;padding:6px 10px;"><b>Weight / Gradient</b></td></tr>
      <tr><td>$W_1$</td><td>up-proj 权重（含 gate + up）</td><td>$[E, 2I, d]$ = 3 GB</td></tr>
      <tr><td>$W_2$</td><td>down-proj 权重</td><td>$[E, d, I]$ = 1.5 GB</td></tr>
      <tr><td>$dO$</td><td>上游梯度</td><td>$[T, d]$ BF16</td></tr>
      <tr><td>$dA'$</td><td>$dO \cdot W_2^\top$（在 TMEM 内）</td><td>$[TK, I]$ —— 永不落 HBM</td></tr>
      <tr><td>$dH$</td><td>pre-activation gradient</td><td>$[TK, 2I]$ BF16 = 1.5 GB</td></tr>
      <tr><td>$dS$</td><td>router score gradient</td><td>$[T, K]$ FP32 —— 从 $\langle dA', A\rangle$ 行内归约得到</td></tr>
      <tr><td>$dW_1, dW_2$</td><td>权重梯度（varlen-K Grouped GEMM 输出）</td><td>同 $W_1, W_2$ shape，FP32</td></tr>
      <tr><td colspan="3" style="background:#fafafa;padding:6px 10px;"><b>Routing metadata（SonicMoE 专用）</b></td></tr>
      <tr><td><code>x_gather_idx</code></td><td>grouped 位置 → 原始 token id</td><td>$[TK]$ int32，喂 TMA gather4 的 A_idx</td></tr>
      <tr><td><code>s_scatter_idx</code></td><td>grouped → $s$ flatten 后的下标</td><td>$[TK]$ int32</td></tr>
      <tr><td><code>s_reverse_scatter_idx</code></td><td>反向映射，把 $y$ 写回 $O$</td><td>$[TK]$ int32</td></tr>
      <tr><td><code>expert_frequency_offset</code></td><td>exclusive prefix-sum of $M_e$（即 <code>cu_seqlens_m</code>）</td><td>$[E+1]$ int32</td></tr>
    </tbody>
  </table>

  <p class="prologue-foot"><b>阅读提示：</b>后面"深度解读"块里出现的所有数字（HBM 流量 / 显存占用 / cache hit / 2GB / 1.5GB 等）都基于以上 Qwen3-235B-A22B 配置，便于横向比较。</p>
</section>
 <d-contents> <nav class="l-text figcaption"> <h3>Contents</h3>
<h3 class="zh-h">目录</h3> <div> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#1-opportunities-and-pains-of-fine-grained-moes">1. Opportunities and Pains of Fine-Grained MoEs</a> </div> <ul> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#moe-as-grouped-gemm">MoE as Grouped GEMM</a> </li> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#sonicmoe-the-algorithm-and-kernel-decomposition">SonicMoE - the Algorithm and Kernel Decomposition</a> </li> </ul> <div> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#2-the-software-abstraction-of-quack-that-empowers-sonicmoe">2. the Software Abstraction of QuACK that Empowers SonicMoE</a> </div> <ul> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#tiled-gemm-kernel-on-nvidia-gpus">Tiled GEMM kernel on NVIDIA GPUs</a> </li> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#customizable-epilogue">Customizable Epilogue</a> </li> </ul> <div> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#3-underneath-the-abstraction-hardware-features-that-empower-the-io-overlap">3. Underneath the Abstraction - Hardware Features that Empower the IO Overlap</a> </div> <ul> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#gemm-programming-model">GEMM programming model</a> </li> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#2cta-mma">2CTA MMA</a> </li> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#native-dynamic-persistent-tile-scheduler">Native Dynamic Persistent Tile Scheduler</a> </li> </ul> <div> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#4-reducing-the-impact-of-io-costs">4. Reducing the Impact of IO Costs</a> </div> <ul> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#gather-fusion">Gather Fusion</a> </li> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#swiglu-and-dswiglu-fusion">SwiGLU and dSwiGLU Fusion</a> </li> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#overlapping-io-with-mma-compute-dh-kernel">Overlapping IO with MMA Compute - dH kernel</a> </li> </ul> <div> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#5-benchmark-results">5. Benchmark Results</a> </div> <ul> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#forward-and-backward-tflops-of-6-open-source-moe-configs">Forward and Backward TFLOPS of 6 Open-source MoE Configs</a> </li> <li> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#profiling-time-breakdown">Profiling Time Breakdown</a> </li> </ul> <div> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#conclusion">Conclusion</a> </div> <div> <a href="https://dao-lab.ai/blog/2026/sonicmoe-blackwell/#appendix">Appendix</a> </div> </nav> </d-contents> <style>.post img{max-width:100%;height:auto}.post blockquote p{margin-top:.2em;margin-bottom:.2em;line-height:1.4}.post blockquote p:first-child{margin-top:0}.post blockquote p:nth-child(2),.post blockquote p:nth-child(3){margin-bottom:1em}.post blockquote strong{font-style:normal!important}.post blockquote{background-color:rgba(76,158,255,0.08);border:1px solid var(--global-theme-color,#4c9eff);border-left:4px solid var(--global-theme-color,#4c9eff);border-radius:4px;padding:1rem 1.5rem;font-size:inherit;color:inherit;max-width:85%;margin:1.5rem auto}.post blockquote .MJXc-display,.post blockquote .katex-display{text-align:center!important;margin:1em 0!important}.post blockquote .MathJax,.post blockquote .katex,.post blockquote .MathJax_Display,.post blockquote mjx-container,.post blockquote mjx-math,.post blockquote mjx-mrow,.post blockquote .MathJax *,.post blockquote mjx-container *{color:inherit!important}html[data-theme='dark'] .post blockquote .MathJax,html[data-theme='dark'] .post blockquote mjx-container,html[data-theme='dark'] .post blockquote mjx-container *{color:var(--global-text-color)!important}.post h1{font-weight:normal!important;font-style:normal!important;border-bottom:1px solid var(--global-divider-color)!important;padding-bottom:.5rem!important}.post h1{margin-top:3rem!important;margin-bottom:1.5rem!important}.post h2{margin-top:2.5rem!important;margin-bottom:1.25rem!important}.post h3{margin-top:2rem!important;margin-bottom:1rem!important}.post h4{margin-top:1.5rem!important;margin-bottom:.75rem!important}.post h5,.post h6{margin-top:1rem!important;margin-bottom:.5rem!important}</style> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/blogpost_teasor.png" width="100%"></p> <p align="center"><em>Figure: SonicMoE's per-layer activation memory footprint (left) stays constant even when expert granularity (embedding dimension / expert intermediate dimension) increases, and SonicMoE can achieve 1.87-4.04x relative speedup compared to existing MoE training kernels ScatterMoE and MoMoE. </em></p>
<p class="zh-tr" align="center">图：SonicMoE 的单层 activation memory 占用（左）即使在 expert granularity（embedding dimension / expert intermediate dimension）增加时仍保持常数，且相对于现有 MoE 训练 kernel ScatterMoE 与 MoMoE 取得 1.87–4.04× 加速。</p> <p><strong>SonicMoE now runs at peak throughput on NVIDIA Blackwell GPUs (B200/B300), in addition to its existing Hopper (H100) support.</strong> This blogpost walks through how we got there: an IO-aware algorithm that keeps activation memory independent of expert granularity, a unified software abstraction on <a href="https://github.com/Dao-AILab/quack" rel="external nofollow noopener" target="_blank">QuACK</a> that makes porting across GPU architectures straightforward, and the Blackwell hardware features we exploit to hide IO costs behind computation.</p>
<p class="zh-tr">SonicMoE 现已在 NVIDIA Blackwell GPU（B200/B300）上达到峰值吞吐，同时继续支持 Hopper（H100）。本博客讲述这一过程：一个让 activation memory 与 expert granularity 解耦的 IO-aware 算法、一个让跨 GPU 架构移植变得简单的 QuACK 软件抽象、以及我们利用的 Blackwell 硬件特性，把 IO 成本藏到计算后面。</p> <p align="center"> <a href="https://arxiv.org/abs/2512.14080" rel="external nofollow noopener" target="_blank"><img src="https://img.shields.io/badge/arXiv-2512.14080-b31b1b.svg" alt="arXiv"></a> <a href="https://github.com/Dao-AILab/sonic-moe" rel="external nofollow noopener" target="_blank"><img src="about:blank" alt="Code"></a> <a href="https://pypi.org/project/sonic-moe/" rel="external nofollow noopener" target="_blank"><img src="about:blank" alt="PyPI"></a> </p> <h2 id="1-opportunities-and-pains-of-fine-grained-moes">1. Opportunities and Pains of Fine-Grained MoEs</h2>
<h2 class="zh-h" id="1-opportunities-and-pains-of-fine-grained-moes">1. Fine-Grained MoE 的机会与代价</h2> <p>Mixture-of-Experts (MoE) models have become the dominant architecture for scaling language models without proportionally increasing compute. The appeal is straightforward: by routing each token to a small subset of <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="0" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi></math></mjx-assistive-mml></mjx-container> out of <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="1" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D438 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>E</mi></math></mjx-assistive-mml></mjx-container> expert networks, we get a model with hundreds of billions of parameters at the compute cost of a much smaller dense model. The training FLOP savings and quality improvements are well-established, but they come with hardware costs that grow worse as models become more fine-grained.</p>
<p class="zh-tr">Mixture-of-Experts (MoE) 模型已成为在不等比例增加 compute 的前提下扩展语言模型的主流架构。其吸引力很直接：把每个 token 路由到 E 个 expert 中的 K 个，就能用一个小得多的 dense 模型的算力跑出一个数千亿参数的总模型。训练 FLOP 的节省与 quality 提升早已证实，但代价是当模型变得更 fine-grained 时，硬件成本会越来越糟。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/finegrained-MoE.png" width="70%"></p> <p align="center"><em>Figure: fine-grained MoE architecture [1] </em></p>
<p class="zh-tr" align="center">图：fine-grained MoE 架构 [1]</p> <p>Two architectural dimensions define how an MoE model trades off quality and efficiency.</p>
<p class="zh-tr">用两个架构维度刻画 MoE 在质量和效率之间的取舍：</p> <ul> <li> <p><strong>Granularity</strong> (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="2" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43A TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="4"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-texatom texclass="ORD"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2F"></mjx-c></mjx-mo></mjx-texatom><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D45B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>G</mi><mo>=</mo><mi>d</mi><mrow data-mjx-texclass="ORD"><mo>/</mo></mrow><mi>n</mi></math></mjx-assistive-mml></mjx-container>, where <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="3" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi></math></mjx-assistive-mml></mjx-container> is the model embedding dimension and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="4" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D45B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math></mjx-assistive-mml></mjx-container> is each expert’s intermediate size) measures how small the experts are relative to the model width. A high-granularity (fine-grained) MoE has many small experts.</p>
<p class="zh-tr">Granularity（$G = d/n$，$d$ 是模型 embedding dimension，$n$ 是单个 expert 的 intermediate size）衡量 expert 相对模型宽度有多小。高 granularity（fine-grained）的 MoE 拥有许多小 expert。</p> </li> <li> <p><strong>Sparsity</strong> (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="5" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="4"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-texatom texclass="ORD"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2F"></mjx-c></mjx-mo></mjx-texatom><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D438 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ρ</mi><mo>=</mo><mi>K</mi><mrow data-mjx-texclass="ORD"><mo>/</mo></mrow><mi>E</mi></math></mjx-assistive-mml></mjx-container>) measures the ratio of experts activated per token.</p>
<p class="zh-tr">Sparsity（$\rho = K/E$）衡量每个 token 激活 expert 的比例。</p> </li> </ul> <p>MoE scaling laws, from controlled experiments (e.g. <a href="https://arxiv.org/pdf/2402.07871" rel="external nofollow noopener" target="_blank">Krajewski et al.</a> and <a href="https://arxiv.org/pdf/2507.17702" rel="external nofollow noopener" target="_blank">Tian et al.</a>) and recent open-source model scaling trends, consistently show that higher granularity and higher sparsity yield better model quality per FLOP: selecting more, smaller experts increases representational capacity, while sparser activation allows more total parameters within the same compute budget. Frontier open-source models reflect this clearly: <a href="https://huggingface.co/mistralai/Mixtral-8x22B-v0.1" rel="external nofollow noopener" target="_blank">Mixtral 8x22B</a>, released in 2024, operated at <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="6" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43A TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c33"></mjx-c><mjx-c class="mjx-c38"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>G</mi><mo>=</mo><mn>0.38</mn></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="7" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c32"></mjx-c><mjx-c class="mjx-c35"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ρ</mi><mo>=</mo><mn>0.25</mn></math></mjx-assistive-mml></mjx-container>, while recent models since 2025 like <a href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2" rel="external nofollow noopener" target="_blank">DeepSeek V3.2</a> (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="8" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43A TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c33"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c35"></mjx-c><mjx-c class="mjx-c30"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>G</mi><mo>=</mo><mn>3.50</mn></math></mjx-assistive-mml></mjx-container>, <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="9" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c33"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ρ</mi><mo>=</mo><mn>0.03</mn></math></mjx-assistive-mml></mjx-container>), <a href="https://huggingface.co/moonshotai/Kimi-K2.5" rel="external nofollow noopener" target="_blank">Kimi K2.5</a> (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="10" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43A TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c33"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c35"></mjx-c><mjx-c class="mjx-c30"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>G</mi><mo>=</mo><mn>3.50</mn></math></mjx-assistive-mml></mjx-container>, <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="11" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ρ</mi><mo>=</mo><mn>0.02</mn></math></mjx-assistive-mml></mjx-container>), and <a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct" rel="external nofollow noopener" target="_blank">Qwen3-Next-80B-A3B-Instruct</a> (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="12" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43A TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c34"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c30"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>G</mi><mo>=</mo><mn>4.00</mn></math></mjx-assistive-mml></mjx-container>, <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="13" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c2E"></mjx-c><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ρ</mi><mo>=</mo><mn>0.02</mn></math></mjx-assistive-mml></mjx-container>) have pushed both dimensions aggressively. Every new generation of frontier MoE is more fine-grained and sparser than the last.</p>
<p class="zh-tr">MoE scaling laws —— 来自 controlled experiment（如 Krajewski 等、Tian 等）以及近期开源模型的 scaling 趋势 —— 一致表明：更高的 granularity 与更高的 sparsity 在等 FLOP 下带来更好的模型质量；选择更多更小的 expert 提升 representational capacity，更稀疏的激活又允许在同等 compute 预算下塞更多总参数。前沿开源模型清晰地反映了这一点：2024 年的 Mixtral 8×22B 跑在 $G=0.38, \rho=0.25$；2025 年以来的 DeepSeek V3.2（$G=3.50,\rho=0.03$）、Kimi K2.5（$G=3.50,\rho=0.02$）、Qwen3-Next-80B-A3B-Instruct（$G=4.00,\rho=0.02$）则把两个维度都激进推高。每一代前沿 MoE 都比上一代更 fine-grained、更稀疏。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>为什么 fine-grained 比 dense 还省 FLOPs</strong>
<p>一个 K-of-E MoE 层的 forward FLOPs 是 $6\,T\,n\,K\,d$（up-proj + down-proj，等于 dense MLP $6\,T\,d\,(nK)$）。<b>等 FLOPs 约束下 $nK$ 是常数</b>，所以"granularity 提升"在实际 scaling 实验里意味着 $n$ 减小、同时 $K$ 等比例增加。</p>
<p>这带来两个同时发生的效应：(i) 更多更小的 expert 提升 representational capacity（Krajewski 等的 scaling law 证实）；(ii) <b>任何 $O(TKd)$ 的中间张量在等 FLOPs 下线性长大</b>，因为 $K$ 涨了。SonicMoE 的算法贡献就是把 (ii) 这条曲线压扁。</p>
<p>另一个少被提到的点：$\rho$ 减小让单 expert 平均收到的 token 数 $T\rho$ 变小，直接刺穿 GEMM 的 $M$ 维度，让每个 expert 的 GEMM 从"高瘦"变成"瘦到 Tensor Core 吃不饱"。</p>
</div> <p>However, the pursuit of granularity and sparsity comes with two painful hardware costs:</p>
<p class="zh-tr">然而追求 granularity 与 sparsity 带来两个棘手的硬件成本：</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/act-mem-io-vs-granularity.png" width="90%"></p> <p align="center"><em>Figure: Per-layer activation memory (left) and forward IO costs (right) as expert granularity increases. We fix microbatch size as 32768 and each model's embedding dimension, then vary the expert intermediate size while keeping training FLOPs and parameter count constant. </em></p>
<p class="zh-tr" align="center">图：随 expert granularity 增大，单层 activation memory（左）与前向 IO 成本（右）的变化。固定 microbatch=32768 与各模型的 embedding dimension，仅改变 expert intermediate size，同时保持 training FLOPs 与参数量不变。</p> <h4 id="problem-1-activation-memory-scales-with-expert-granularity-with-current-training-kernels">Problem 1: Activation Memory Scales with Expert Granularity with Current Training Kernels.</h4>
<h4 class="zh-h" id="problem-1-activation-memory-scales-with-expert-granularity-with-current-training-kernels">Problem 1：现行训练 kernel 下 Activation Memory 随 expert granularity 线性增长</h4> <p>During training, intermediate tensors must be cached for the backward pass. The total FLOPs of MoE forward and backward computation is <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="14" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mn class="mjx-n"><mjx-c class="mjx-c36"></mjx-c></mjx-mn><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-c2B"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="3"><mjx-c class="mjx-c31"></mjx-c><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D45B TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo stretchy="false">(</mo><mn>6</mn><mo>+</mo><mn>12</mn><mo stretchy="false">)</mo><mi>T</mi><mi>n</mi><mi>K</mi><mi>d</mi></math></mjx-assistive-mml></mjx-container>. For fixed <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="15" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>T</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="16" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi></math></mjx-assistive-mml></mjx-container>, keeping FLOPs constant requires <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="17" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D45B TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi><mi>K</mi></math></mjx-assistive-mml></mjx-container> to stay constant. Increasing granularity means decreasing <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="18" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D45B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math></mjx-assistive-mml></mjx-container> and proportionally increasing <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="19" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi></math></mjx-assistive-mml></mjx-container>. Any activation of size <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="20" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>O</mi><mo stretchy="false">(</mo><mi>T</mi><mi>K</mi><mi>d</mi><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container> thus grows linearly with granularity.</p>
<p class="zh-tr">训练时中间张量必须为反向缓存。MoE 前后向计算的总 FLOPs 是 $(6+12)TnKd$。固定 $T$ 与 $d$ 时，要等 FLOPs 就必须 $nK$ 不变；granularity 增大意味着 $n$ 减小、$K$ 等比例增加。任何大小为 $O(TKd)$ 的 activation 都因此随 granularity 线性增长。</p> <p>For current MoE kernels like <a href="https://arxiv.org/pdf/2403.08245" rel="external nofollow noopener" target="_blank">ScatterMoE</a> and <a href="https://github.com/tilde-research/MoMoE-impl" rel="external nofollow noopener" target="_blank">MoMoE</a>, variables such as the down-proj output <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="21" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> (size <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="22" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>T</mi><mi>K</mi><mi>d</mi></math></mjx-assistive-mml></mjx-container>) are cached for the backward pass, causing per-layer activation memory to grow linearly as experts become more fine-grained. Prior solutions such as MoMoE usually require a GEMM recomputation during backward to trade off activation memory for extra FLOPs. This raises the question:</p>
<p class="zh-tr">对当前 MoE kernel（如 ScatterMoE 与 MoMoE），down-proj 输出 $Y$（大小 $TKd$）等变量被为反向缓存，导致 single layer 的 activation memory 随 expert 变细而线性增大。MoMoE 等先前方案通常需要在反向重算 GEMM 来用额外 FLOPs 换 activation memory。这促使我们提出问题：</p> <p align="center"><em>Is it possible to achieve activation memory efficiency without extra FLOPs from GEMM recomputation?</em></p>
<p class="zh-tr" align="center">在不引入 GEMM recomputation 额外 FLOPs 的前提下，能否实现 activation memory 高效？</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>等 FLOPs 约束下 cache 张量随 $K$ 如何增长</strong>
<p>工业界做 MoE scaling 实验通常固定训练 FLOPs 与 activated 参数量 —— 即 $T, d, nK$ 是常数，只调 $G=d/n$ 和 $K$。下表给出 Qwen3-235B-A22B 单层、$T=32k$、$d=4096$ 下三种 cache 张量随 $K$ 的增长：</p>
<table style="border-collapse:collapse;font-size:13px;margin:8px 0;">
<thead><tr><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">$K$</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">$X_g$ ($TKd$ BF16)</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">$Y$</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">$Y_\text{scattered}$</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">合计</th></tr></thead>
<tbody>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">2</td><td style="padding:4px 10px;border:1px solid #ccc;">512 MB</td><td style="padding:4px 10px;border:1px solid #ccc;">512 MB</td><td style="padding:4px 10px;border:1px solid #ccc;">512 MB</td><td style="padding:4px 10px;border:1px solid #ccc;">1.5 GB</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">4</td><td style="padding:4px 10px;border:1px solid #ccc;">1 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">1 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">1 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">3 GB</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;"><b>8（Qwen3-235B）</b></td><td style="padding:4px 10px;border:1px solid #ccc;">2 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">2 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">2 GB</td><td style="padding:4px 10px;border:1px solid #ccc;"><b>6 GB</b></td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">16</td><td style="padding:4px 10px;border:1px solid #ccc;">4 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">4 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">4 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">12 GB</td></tr>
</tbody></table>
<p>× 94 层 ⇒ Qwen3-235B 全模型多出几百 GB activation。SonicMoE 要做的就是把这整张表的"合计"列压到 0。</p>
<p>顺带：MoMoE 反向重算 down-proj GEMM，<b>不是 element-wise 而是真正的 Tensor Core GEMM</b>，反向时间约 +20%。而 SonicMoE 声称"无额外 FLOPs"的精确含义是：它只在反向 epilogue inline 重算一个 element-wise 的 $\mathrm{SwiGLU}(h)$ —— 几乎免费。</p>
</div> <h4 id="problem-2-io-cost-scales-with-expert-granularity-and-moe-sparsity">Problem 2: IO Cost Scales with Expert Granularity and MoE Sparsity.</h4>
<h4 class="zh-h" id="problem-2-io-cost-scales-with-expert-granularity-and-moe-sparsity">Problem 2：IO Cost 随 expert granularity 与 MoE sparsity 同时恶化</h4> <p>A GPU kernel’s runtime is determined by whichever resource is exhausted first: compute throughput (FLOP/s) or memory bandwidth (bytes/s). <strong>Arithmetic intensity as the ratio of FLOPs to HBM bytes transferred is the metric that determines in which regime a kernel operates.</strong> As the arithmetic intensity becomes higher, the kernel is likely to be compute-bound rather than memory-bound.</p>
<p class="zh-tr">GPU kernel 的 runtime 由先撑爆的资源决定：compute throughput（FLOP/s）或 memory bandwidth（bytes/s）。算术强度（FLOPs / HBM bytes）是判断 kernel 落在哪个 regime 的指标。算术强度越高，kernel 越可能 compute-bound 而不是 memory-bound。</p> <p>Assuming perfect load balancing and SwiGLU activation, the arithmetic intensity of a single expert’s forward pass is lower-bounded by:</p>
<p class="zh-tr">假设完美 load balancing 与 SwiGLU activation，单 expert 前向的算术强度下界为：</p><span> <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" display="true" tabindex="0" ctxtmenu_counter="23" style="font-size: 119.4%; position: relative;"><mjx-math display="true" class="MJX-TEX" aria-hidden="true" style="margin-left: 0px; margin-right: 0px;"><mjx-mtext class="mjx-n"><mjx-c class="mjx-c41"></mjx-c><mjx-c class="mjx-c72"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c68"></mjx-c><mjx-c class="mjx-c6D"></mjx-c><mjx-c class="mjx-c65"></mjx-c><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c63"></mjx-c><mjx-c class="mjx-c20"></mjx-c><mjx-c class="mjx-c49"></mjx-c><mjx-c class="mjx-c6E"></mjx-c><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c65"></mjx-c><mjx-c class="mjx-c6E"></mjx-c><mjx-c class="mjx-c73"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c79"></mjx-c></mjx-mtext><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mfrac space="4"><mjx-frac type="d"><mjx-num><mjx-nstrut type="d"></mjx-nstrut><mjx-mn class="mjx-n"><mjx-c class="mjx-c33"></mjx-c></mjx-mn></mjx-num><mjx-dbox><mjx-dtable><mjx-line type="d"></mjx-line><mjx-row><mjx-den><mjx-dstrut type="d"></mjx-dstrut><mjx-mrow><mjx-mfrac><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-c2B"></mjx-c></mjx-mo><mjx-mfrac space="3"><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mrow size="s"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43A TEX-I"></mjx-c></mjx-mi></mjx-mrow></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-c2B"></mjx-c></mjx-mo><mjx-mfrac space="3"><mjx-frac><mjx-num><mjx-nstrut></mjx-nstrut><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c33"></mjx-c></mjx-mn></mjx-num><mjx-dbox><mjx-dtable><mjx-line></mjx-line><mjx-row><mjx-den><mjx-dstrut></mjx-dstrut><mjx-mrow size="s"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi></mjx-mrow></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac></mjx-mrow></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="4"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-mrow space="2"><mjx-mo class="mjx-s3"><mjx-c class="mjx-c28 TEX-S3"></mjx-c></mjx-mo><mjx-mo class="mjx-n"><mjx-c class="mjx-c6D"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6E"></mjx-c></mjx-mo><mjx-mrow space="2"><mjx-mo class="mjx-s3"><mjx-c class="mjx-c28 TEX-S3"></mjx-c></mjx-mo><mjx-mfrac><mjx-frac type="d"><mjx-num><mjx-nstrut type="d"></mjx-nstrut><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-num><mjx-dbox><mjx-dtable><mjx-line type="d"></mjx-line><mjx-row><mjx-den><mjx-dstrut type="d"></mjx-dstrut><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43A TEX-I"></mjx-c></mjx-mi></mjx-den></mjx-row></mjx-dtable></mjx-dbox></mjx-frac></mjx-mfrac><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="2"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-s3"><mjx-c class="mjx-c29 TEX-S3"></mjx-c></mjx-mo></mjx-mrow><mjx-mo class="mjx-s3"><mjx-c class="mjx-c29 TEX-S3"></mjx-c></mjx-mo></mjx-mrow></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtext>Arithmetic Intensity</mtext><mo>=</mo><mfrac><mn>3</mn><mrow><mfrac><mn>2</mn><mi>d</mi></mfrac><mo>+</mo><mfrac><mrow><mn>2</mn><mi>G</mi></mrow><mi>d</mi></mfrac><mo>+</mo><mfrac><mn>3</mn><mrow><mi>T</mi><mi>ρ</mi></mrow></mfrac></mrow></mfrac><mo>=</mo><mi>O</mi><mrow data-mjx-texclass="INNER"><mo data-mjx-texclass="OPEN">(</mo><mo data-mjx-texclass="OP" movablelimits="true">min</mo><mrow data-mjx-texclass="INNER"><mo data-mjx-texclass="OPEN">(</mo><mfrac><mi>d</mi><mi>G</mi></mfrac><mo>,</mo><mi>T</mi><mi>ρ</mi><mo data-mjx-texclass="CLOSE">)</mo></mrow><mo data-mjx-texclass="CLOSE">)</mo></mrow></math></mjx-assistive-mml></mjx-container> </span><p>where <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="24" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>T</mi></math></mjx-assistive-mml></mjx-container> is the number of tokens in a microbatch (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="25" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>T</mi><mi>ρ</mi></math></mjx-assistive-mml></mjx-container> is the average number of routed tokens per expert).</p>
<p class="zh-tr">其中 $T$ 是 microbatch 内 token 数（$T\rho$ 是平均每 expert 的 routed token 数）。</p> <p>In this case, <strong>both increasing <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="26" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43A TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>G</mi></math></mjx-assistive-mml></mjx-container> and increasing MoE sparsity (decreasing <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="27" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D70C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ρ</mi></math></mjx-assistive-mml></mjx-container>) would drive arithmetic intensity down.</strong> For example, <a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct" rel="external nofollow noopener" target="_blank">Qwen3-Next-80B-A3B-Instruct</a> would have an arithmetic intensity of 210 for a microbatch of 16K tokens, while an iso-param dense SwiGLU MLP would have an arithmetic intensity of 2570, 12× higher. In this regime, kernel runtime is dominated by the IO costs, not compute throughput.</p>
<p class="zh-tr">在这种情况下，granularity $G$ 增大与 sparsity 增大（$\rho$ 减小）都会把算术强度推下去。例如 Qwen3-Next-80B-A3B-Instruct 在 16K microbatch 下算术强度仅 210，而同等参数的 dense SwiGLU MLP 是 2570（高 12×）。在这个 regime 里，kernel runtime 由 IO cost 主导，而非 compute throughput。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>AI 下界公式怎么来的 —— 值得自己推一次</strong>
<p>分子 3 = forward + backward-act + backward-weight 三次 GEMM 共享同一份 activation 的粗略 FLOP/byte 系数。分母三项的含义：</p>
<ul>
<li>$\frac{2}{d}$：weight 从 HBM 读到 SMEM 的字节摊到 FLOPs 上（每条 weight 边 $d$ 维）。</li>
<li>$\frac{2G}{d} = \frac{2}{n}$：activation 在 expert 内部的字节摊到 FLOPs 上（受 $n$ 控制）。$G$ 越大 ⇒ $n$ 越小 ⇒ 这一项越大 ⇒ 分母大 ⇒ AI 小。</li>
<li>$\frac{3}{T\rho}$：每个 expert 平均收到 $T\rho$ token，token 太少 ⇒ 一份 weight 摊不出多少 FLOP ⇒ AI 退化。</li>
</ul>
<p>口诀：<b>"$d$ 大、$n$ 大、$T\rho$ 大"才进 compute-bound</b>。MoE 把 $n$ 故意做小（fine-grained）、$T\rho$ 故意做小（sparse），这两条都把 kernel 推向 memory-bound。</p>
<p>工程后果：B300 的 BF16 算力约 2.5 PFLOPs，HBM 7.7 TB/s。AI 临界值约 $2.5\text{P}/7.7\text{T} \approx 325$。Qwen3-Next 的 210 < 325 ⇒ <b>HBM 决定 runtime，kernel 优化的核心变成"少读少写 HBM"</b>而不是"让 Tensor Core 更忙"。后面所有 fusion 都围绕"消灭 $O(TKd)$ HBM 往返"展开，都是这条逻辑推出来的。</p>
<p>EP 延伸：footnote 的意思是 NVLink 0.9 TB/s vs HBM 7.7 TB/s 慢 8×，IB 0.4 TB/s 慢 19× ⇒ IO-aware 设计在 expert parallelism 下更关键。</p>
</div> <blockquote> <p>For fine-grained and sparse MoEs, every expert’s GEMM problem shape is small enough such that the kernel falls into the memory-bound regime.</p>
<p class="zh-tr">对于 fine-grained 与 sparse MoE，每个 expert 的 GEMM 形状都小到 kernel 落入 memory-bound regime。</p> </blockquote> <p><strong>These IO costs will become a greater bottleneck in expert parallelism, as the intra- or inter-node network bandwidth are often <em>much</em> slower than HBM loading speed.</strong> SonicMoE currently focuses on the case of single GPU (EP degree=1), but the IO-aware algorithmic designs are transferable to expert parallelism.</p>
<p class="zh-tr">在 expert parallelism 下这些 IO cost 会成为更大的瓶颈，因为 intra-/inter-node 网络带宽通常远低于 HBM。SonicMoE 当前聚焦单 GPU（EP degree=1），但 IO-aware 算法设计原则可迁移到 expert parallelism。</p> <h3 id="moe-as-grouped-gemm">MoE as Grouped GEMM</h3>
<h3 class="zh-h" id="moe-as-grouped-gemm">MoE as Grouped GEMM</h3> <p>MoE computation is often implemented using Grouped GEMM. A Grouped GEMM is a batch of matrix multiplications with possibly different problem shapes. Following standard BLAS conventions used by CUTLASS, each GEMM computes <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="28" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D436 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="4"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>C</mi><mo>=</mo><mi>A</mi><mi>B</mi></math></mjx-assistive-mml></mjx-container> where <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="29" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c2208"></mjx-c></mjx-mo><mjx-msup space="4"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-ds mjx-b"><mjx-c class="mjx-c211D TEX-A"></mjx-c></mjx-mi></mjx-texatom><mjx-script style="vertical-align: 0.363em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi><mo>∈</mo><msup><mrow data-mjx-texclass="ORD"><mi mathvariant="double-struck">R</mi></mrow><mrow data-mjx-texclass="ORD"><mi>M</mi><mo>×</mo><mi>K</mi></mrow></msup></math></mjx-assistive-mml></mjx-container> (activations), <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="30" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c2208"></mjx-c></mjx-mo><mjx-msup space="4"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-ds mjx-b"><mjx-c class="mjx-c211D TEX-A"></mjx-c></mjx-mi></mjx-texatom><mjx-script style="vertical-align: 0.363em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>B</mi><mo>∈</mo><msup><mrow data-mjx-texclass="ORD"><mi mathvariant="double-struck">R</mi></mrow><mrow data-mjx-texclass="ORD"><mi>K</mi><mo>×</mo><mi>N</mi></mrow></msup></math></mjx-assistive-mml></mjx-container> (weights), and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="31" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D436 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c2208"></mjx-c></mjx-mo><mjx-msup space="4"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-ds mjx-b"><mjx-c class="mjx-c211D TEX-A"></mjx-c></mjx-mi></mjx-texatom><mjx-script style="vertical-align: 0.363em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>C</mi><mo>∈</mo><msup><mrow data-mjx-texclass="ORD"><mi mathvariant="double-struck">R</mi></mrow><mrow data-mjx-texclass="ORD"><mi>M</mi><mo>×</mo><mi>N</mi></mrow></msup></math></mjx-assistive-mml></mjx-container> (outputs).</p>
<p class="zh-tr">MoE 计算通常用 Grouped GEMM 实现 —— 一批形状可能不同的矩阵乘。沿用 CUTLASS 的 BLAS 约定，每个 GEMM 是 $C = AB$，$A \in \mathbb{R}^{M\times K}$ 是 activation，$B\in\mathbb{R}^{K\times N}$ 是 weight，$C\in\mathbb{R}^{M\times N}$ 是 output。</p> <p>In MoE, each expert usually receives a different number of tokens, and input tokens may need to be gathered from different positions, or they may already be contiguously packed by expert.</p>
<p class="zh-tr">在 MoE 中每个 expert 通常收到不同数量的 token，输入 token 可能需要从不同位置 gather，也可能已经按 expert 连续打包好。</p> <p>For the forward pass and backward activation gradient, we would need Grouped GEMM with input shapes that have constant <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="32" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>N</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="33" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi></math></mjx-assistive-mml></mjx-container> (embedding dimension and expert intermediate dimension) but different <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="34" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>M</mi></math></mjx-assistive-mml></mjx-container> (the number of routed tokens per expert). <strong>We call this varlen-M Grouped GEMM</strong>. (CUTLASS would describe it as <em>Grouped GEMM with ragged M dimensions</em>). For the backward weight gradient, we would reduce over token embeddings for each expert GEMM, in which <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="35" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>M</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="36" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>N</mi></math></mjx-assistive-mml></mjx-container> (embedding dimension and expert intermediate dimension) are fixed but the <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="37" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi></math></mjx-assistive-mml></mjx-container> dimension varies. <strong>We call this varlen-K Grouped GEMM</strong>.</p>
<p class="zh-tr">前向与反向激活梯度需要 $N$ 与 $K$ 固定（embedding dimension、expert intermediate dimension）但 $M$（每 expert 的 routed token 数）变长的 Grouped GEMM —— 我们称为 varlen-M Grouped GEMM（CUTLASS 称为 "Grouped GEMM with ragged M dimensions"）。反向权重梯度需要在 token embedding 维度上 reduce，$M$ 与 $N$（embedding dimension、expert intermediate dimension）固定但 $K$ 变长 —— 我们称为 varlen-K Grouped GEMM。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>varlen-M / varlen-K 的硬件含义</strong>
<p><b>varlen-M</b>：每个 expert 的 $M_e$ 不同，tile scheduler 给 CTA 分配 tile 时必须按 expert 切；同一 expert 的 tile 共享同一份 $W$（CTA 在同一 expert 内执行 persistent loop 可以最大化 $W$ 在 L2 的复用）。CUTLASS 用 <code>cu_seqlens_m</code>（exclusive prefix-sum）传 $M_e$ 边界。</p>
<p><b>varlen-K</b>：要在 token 维度做 reduction，K-dim 长度不一 —— GEMM 内循环长度按 expert 变化，split-K 不再适用（每 expert 独立做 reduction），通常用 persistent kernel + per-expert prologue。</p>
<p>SonicMoE 在 <code>sonicmoe/functional/forward.py:107</code> 调 <code>gemm(... cu_seqlens_m=expert_frequency_offset ...)</code>，在 <code>backward.py:225</code> 调 <code>gemm(... cu_seqlens_k=expert_frequency_offset ...)</code> —— 同一份 QuACK <code>gemm</code> API 通过参数切换两种 ragged 模式，背后是 tile_scheduler 区分 varlen-M / varlen-K 的代码路径。</p>
</div> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/input-formats.png" width="36%" style="margin-right: 50px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/grouped-gemm.png" width="47%"></p>
<p class="zh-tr" align="center">图：左 — 每个 expert 从输入 tensor 的不同位置 gather 输入（上），或从一个分组好的输入数组读连续段（下）。右 — Grouped GEMM 在 MoE 中的使用示意。</p> <p align="center"><em>Left: Each expert gathers inputs from different positions on an input tensor (top) or reads a contiguous chunk on a grouped input array (bottom). Right: Illustration of using Grouped GEMM in MoE.</em></p>
<p class="zh-tr" align="center">下面用 varlen-M Grouped GEMM 构造一个标准 MoE 前向 pass：</p> <p>We can use varlen-M Grouped GEMM to build a standard MoE forward pass as demonstrated in the following code snippet.</p>
<p class="zh-tr">图：标准 PyTorch MoE forward pass 的可视化 workflow（左）与对应的参考代码（右）。每条黄色虚线标记一次 kernel 边界。标准实现会启动 6 个独立 kernel：gather、up-proj Grouped GEMM、SwiGLU、down-proj Grouped GEMM、scatter、expert aggregation。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/standard-illustration.png" width="100%"></p> <p align="center"><em>Figure: Visual workflow (left) with corresponding reference code (right) of standard MoE forward pass in PyTorch. Each yellow dashed line marks a kernel boundary. The standard implementation launches 6 separate kernels: gather, up-proj Grouped GEMM, SwiGLU, down-proj Grouped GEMM, scatter, and expert aggregation.</em></p>
<p class="zh-tr" align="center">可简化为下面的 workflow 图：</p> <p>This can be simplified to the following workflow diagram:</p>
<p class="zh-tr">图：标准 MoE 实现 forward pass 的 workflow。$\pi$ 是存储 routing metadata 的 binary mask；黄色框是 kernel 边界，蓝色框是 HBM 中的变量，红色 label 标出在前后向之间被 cache 的 activation，紫色框是最终输出。每个变量旁的橙色框按比例代表 Qwen3-235B-A22B-Thinking-2507 MoE 模型在 32k token 下的 tensor 大小。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/standard-workflow-forward.png" width="100%"></p> <p align="center"><em>Figure: Workflow of standard MoE implementation forward pass. π is the binary mask that stores routing metadata. <font color="#fedd26">Yellow boxes</font> are kernel boundaries. <font color="blue">Blue boxes</font> are variables in HBM. <font color="red">Red labels</font> indicate the activations cached across the forward/backward. <font color="purple">Purple boxes</font> are the final outputs. The <font color="orange">orange box</font> beside each variable on global memory represents the tensor size in proportion for Qwen3-235B-A22B-Thinking-2507 MoE model with 32k tokens.</em></p>
<p class="zh-tr" align="center">反向激活梯度的 workflow 就是反向操作，用 dSwiGLU 替换：</p> <p>The workflow of backward activation gradient is simply a reverse operation with dSwiGLU as follows:</p>
<p class="zh-tr">图：标准 MoE 实现 backward activation gradient pass 的 workflow。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/standard-workflow-backward-activation.png" width="100%"></p> <p align="center"><em>Figure: Workflow of standard MoE implementation backward activation gradient pass.</em></p>
<p class="zh-tr" align="center">权重梯度需要用 varlen-K Grouped GEMM 在 token embedding 上 reduce。</p> <p>For weight gradient, we need to use varlen-K Grouped GEMM to reduce over token embeddings.</p>
<p class="zh-tr">图：标准 MoE 实现 backward weight gradient pass 的 workflow。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/standard-workflow-backward-weight.png" width="70%"></p> <p align="center"><em>Figure: Workflow of standard MoE implementation backward weight gradient pass.</em></p>
<p class="zh-tr" align="center">标准实现把每个中间张量都 materialize 到 HBM。这创造了两个都随 expert granularity 增长的代价：</p> <p>The standard implementation materializes every intermediate tensor in HBM between kernel launches. This creates two separate costs that both scale with expert granularity:</p>
<p class="zh-tr">Activation memory：gathered $X$、down-proj 输出 $Y$、scattered $Y$ 都必须为反向缓存，每个占 $2TKd$ 字节。granularity 增大时，这些 $O(TKd)$ 张量线性长大。</p> <ul> <li> <p><strong>Activation memory</strong>: gathered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="38" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container>, down-proj output <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="39" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container>, and scattered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="40" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> must all be cached for the backward pass, each consuming <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="41" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mn>2</mn><mi>T</mi><mi>K</mi><mi>d</mi></math></mjx-assistive-mml></mjx-container> bytes. As granularity increases, these <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="42" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>O</mi><mo stretchy="false">(</mo><mi>T</mi><mi>K</mi><mi>d</mi><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container>-sized tensors grow linearly.</p>
<p class="zh-tr">IO costs：每个 materialize 的中间张量都是一次 HBM round-trip。反向更糟：还要 materialize $dY$ 与 gathered $dO$，都是 $O(TKd)$。fine-grained MoE kernel 跑在 memory-bound regime，IO cost 直接主导 runtime。</p> </li> <li> <p><strong>IO costs</strong>: every materialized intermediate is a round-trip to HBM. The backward pass is worse: it must additionally materialize <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="43" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>Y</mi></math></mjx-assistive-mml></mjx-container> and gathered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="44" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container>, both <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="45" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>O</mi><mo stretchy="false">(</mo><mi>T</mi><mi>K</mi><mi>d</mi><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container>-sized. <strong>Since fine-grained MoE kernels operate in the memory-bound regime, these IO costs directly dominate runtime.</strong></p>
<p class="zh-tr">SonicMoE：算法与 Kernel 分解</p> </li> </ul> <h3 id="sonicmoe-the-algorithm-and-kernel-decomposition">SonicMoE: the Algorithm and Kernel Decomposition</h3>
<h3 class="zh-h" id="sonicmoe-the-algorithm-and-kernel-decomposition">SonicMoE 用一次算法重设计同时解决以上两个问题：我们绕过缓存或 materialize 任何 $O(TKd)$ 大小变量的需求。这让 activation memory 与 expert granularity 解耦，并同时消灭多次主导 runtime 的 HBM 大块往返。</h3> <p><strong>SonicMoE addresses both problems through a single algorithmic redesign: we circumvent the need to cache or materialize any variable with size <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="46" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>O</mi><mo stretchy="false">(</mo><mi>T</mi><mi>K</mi><mi>d</mi><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container>.</strong> This makes activation memory independent of expert granularity, and simultaneously eliminates multiple large HBM round-trips that dominate runtime.</p>
<p class="zh-tr">具体来说，SonicMoE 避免 cache 大小为 $TKd$ 的 down-proj 输出 $Y$、scattered $Y$、gathered $X$；也避免把 $dY$ 与 gathered $dO$ 写到 HBM：</p> <p>In particular, SonicMoE avoids caching down-proj output <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="47" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container>, scattered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="48" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container>, and gathered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="49" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> which all have size <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="50" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>T</mi><mi>K</mi><mi>d</mi></math></mjx-assistive-mml></mjx-container>. We also avoid writing <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="51" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>Y</mi></math></mjx-assistive-mml></mjx-container> and gathered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="52" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container> to HBM:</p>
<p class="zh-tr">Gathered $X$ 与 $dO$：在 kernel runtime 现场 gather，从不 cache gather 结果。</p> <ul> <li> <p><strong>Gathered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="53" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="54" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container></strong>: we gather inputs at each kernel runtime and <em>never</em> cache the gathered results.</p>
<p class="zh-tr">Scattered $Y$：与 aggregation 操作融合 —— 每个 token gather 并求和被激活的 expert 输出。</p> </li> <li> <p><strong>Scattered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="55" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container></strong>: we fuse it with the aggregation operation where each token will gather and sum over activated expert results.</p>
<p class="zh-tr">$Y$ 与 $dY$：重新设计反向计算路径，从 $dO$ 与 $H$ 直接算 $dS$ 与 $dH$，不需要 $Y$ 与 $dY$。先前 MoE kernel（如 ScatterMoE 与 MoMoE）必须为这一步 cache $Y$：</p> </li> <li> <p><strong><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="56" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="57" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>Y</mi></math></mjx-assistive-mml></mjx-container></strong>: we redesign the computational path that starts from <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="58" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="59" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>H</mi></math></mjx-assistive-mml></mjx-container> to directly compute <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="60" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>S</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="61" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> during the backward pass <strong>without <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="62" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="63" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>Y</mi></math></mjx-assistive-mml></mjx-container></strong>. <strong>Prior MoE kernels such as ScatterMoE and MoMoE must cache <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="64" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> for this computation</strong>:</p>
<p class="zh-tr">$dH$：与 $dO$ 做 gather fusion（不需要 $dY$），并用一次额外的 $H$ load 做 dSwiGLU fusion。</p> <ul> <li> <p><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="65" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container>: we apply gather fusion with <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="66" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container> (no need for <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="67" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>Y</mi></math></mjx-assistive-mml></mjx-container>) and dSwiGLU fusion with an extra load of <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="68" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>H</mi></math></mjx-assistive-mml></mjx-container>.</p>
<p class="zh-tr">$dS$：交换 contraction 顺序。等价于把 $S$ 加权放到 down-proj 前向之前，并用 $A$ 与 $dA'$ 计算 $dS$，而不再用 $Y$ 与 $dO$。我们不再需要 cache $Y$。</p> </li> <li> <p><mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="69" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>S</mi></math></mjx-assistive-mml></mjx-container>: we swap the contraction order. <strong>This is equivalent to placing <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="70" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math></mjx-assistive-mml></mjx-container> weighting <em>before</em> down-proj forward pass and using only <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="71" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="72" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: 0.363em;"><mjx-mo class="mjx-var" size="s"><mjx-c class="mjx-c2032"></mjx-c></mjx-mo></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><msup><mi>A</mi><mo data-mjx-alternate="1">′</mo></msup></math></mjx-assistive-mml></mjx-container> for computing <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="73" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>S</mi></math></mjx-assistive-mml></mjx-container> instead of <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="74" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="75" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container>.</strong> We no longer need to cache <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="76" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container>.</p>
<p class="zh-tr">对 expert $e$，记 down-proj 权重为 $W_{2,e}\in\mathbb{R}^{n\times d}$。down-proj 反向激活梯度的 Grouped GEMM 计算 $dA' = dO_e W_2^\top$。</p> <p>For an expert <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="77" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>e</mi></math></mjx-assistive-mml></mjx-container>, denote the down-proj weights for expert <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="78" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>e</mi></math></mjx-assistive-mml></mjx-container> as <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="79" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.104em;"><mjx-texatom size="s" texclass="ORD"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c2208"></mjx-c></mjx-mo><mjx-msup space="4"><mjx-texatom texclass="ORD"><mjx-mi class="mjx-ds mjx-b"><mjx-c class="mjx-c211D TEX-A"></mjx-c></mjx-mi></mjx-texatom><mjx-script style="vertical-align: 0.363em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D45B TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>W</mi><mrow data-mjx-texclass="ORD"><mn>2</mn><mo>,</mo><mi>e</mi></mrow></msub><mo>∈</mo><msup><mrow data-mjx-texclass="ORD"><mi mathvariant="double-struck">R</mi></mrow><mrow data-mjx-texclass="ORD"><mi>n</mi><mo>×</mo><mi>d</mi></mrow></msup></math></mjx-assistive-mml></mjx-container>. The Grouped GEMM in down-proj activation gradient will compute <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="80" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: 0.363em;"><mjx-mo class="mjx-var" size="s"><mjx-c class="mjx-c2032"></mjx-c></mjx-mo></mjx-script></mjx-msup><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="4"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-script></mjx-msub><mjx-msubsup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.288em; margin-left: -0.104em;"><mjx-mi class="mjx-n" size="s" style="margin-left: 0.225em;"><mjx-c class="mjx-c22A4"></mjx-c></mjx-mi><mjx-spacer style="margin-top: 0.18em;"></mjx-spacer><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-script></mjx-msubsup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><msup><mi>A</mi><mo data-mjx-alternate="1">′</mo></msup><mo>=</mo><mi>d</mi><msub><mi>O</mi><mi>e</mi></msub><msubsup><mi>W</mi><mn>2</mn><mi mathvariant="normal">⊤</mi></msubsup></math></mjx-assistive-mml></mjx-container>.</p>
<p class="zh-tr">标准路径计算 $dS_{t,e} = \langle dO_t, Y_{e,t}\rangle$，需要 cache $Y$。代入 $Y_e = A_e W_{2,e}$ 重排 contraction 顺序：</p> <p>The standard path computes <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="81" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.032em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c27E8"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mtext class="mjx-n" space="2"><mjx-c class="mjx-cA0"></mjx-c></mjx-mtext><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.182em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c27E9"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><msub><mi>S</mi><mrow data-mjx-texclass="ORD"><mi>t</mi><mo>,</mo><mi>e</mi></mrow></msub><mo>=</mo><mo fence="false" stretchy="false">⟨</mo><mi>d</mi><msub><mi>O</mi><mi>t</mi></msub><mo>,</mo><mtext>&nbsp;</mtext><msub><mi>Y</mi><mrow data-mjx-texclass="ORD"><mi>e</mi><mo>,</mo><mi>t</mi></mrow></msub><mo fence="false" stretchy="false">⟩</mo></math></mjx-assistive-mml></mjx-container>, which requires caching <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="82" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container>. By substituting <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="83" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.182em;"><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-msub space="4"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-script></mjx-msub><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.104em;"><mjx-texatom size="s" texclass="ORD"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>Y</mi><mi>e</mi></msub><mo>=</mo><msub><mi>A</mi><mi>e</mi></msub><msub><mi>W</mi><mrow data-mjx-texclass="ORD"><mn>2</mn><mo>,</mo><mi>e</mi></mrow></msub></math></mjx-assistive-mml></mjx-container> and rearranging the contraction order:</p>
<p class="zh-tr">$dA'_{e,t}$ 与 $A_{e,t}$ 都不依赖 $dY$ 或 $Y$。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>为什么这一步 G3/G4 baseline 没人做？</strong>
<p>这个换序在数学上只是内积结合律（bit-exact，无任何近似）。能落地需要三件事同时存在：</p>
<ol>
<li><b>fusion 框架支持把"GEMM + 重算 $A$ + dSwiGLU + 行归约 + 三个输出 store"装进同一个 epilogue。</b> ScatterMoE / MoMoE 的 monolithic CUTLASS kernel 没有这种"加几行 lambda"的口子；torch.compile + DeepGEMM 又无法跨 GEMM 边界 fuse。</li>
<li><b>需要一块"长居"的累加器内存放 $dA'$。</b> Hopper 上 WGMMA 累加器分布在 128 线程的 register 里，做行归约要 warp shuffle、register 压力高；Blackwell 的 TMEM（256 KB / SM）可以容纳整个 $[BLK_M, I]$ 的 fp32 累加器，并支持 <code>tcgen05.ld</code> 把任意 sub-tile 拷到 register 做 fusion。</li>
<li><b>需要 epilogue 能同时 store 多个张量而不阻塞 MMA。</b> Hopper 的同步 store 在三 store 串行时拖死 pipeline；Blackwell 的 <code>st.async.release.global</code> 让"一次 epilogue 写 dH/A'/dS 三件" 不会撑爆 critical path。</li>
</ol>
<p>所以 dS 重排不是"想到了"，而是"算法 + 软件抽象 + 硬件原语"三件凑齐之后才能做出来。</p>
<p>源码对应：<code>sonicmoe/functional/backward.py:262-275</code> 一发出 dh/a_prime/ds_scattered 三个输出：</p>
<pre style="background:#1e1e1e;color:#e6e6e6;padding:10px 14px;border-radius:4px;font-size:12px;overflow-x:auto;">_, _, ds_scattered = gemm_dgated(
    dout, w2.permute(2, 0, 1),
    PreAct=h,                     # 反向唯一需要的 forward 缓存
    activation=activation_type,
    dx_out=dh,                    # 输出 #1：dH
    postact_out=a_prime,          # 输出 #2：A 重算（喂 dW2）
    colvec_scale=s,               # 路由权重，行向量
    colvec_reduce=True,           # 输出 #3：行归约 → dS
    cu_seqlens_m=expert_frequency_offset,
    A_idx=x_gather_idx,           # dO 用 TMA gather4
)</pre>
</div> <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" display="true" tabindex="0" ctxtmenu_counter="84" style="font-size: 119.4%; position: relative;"><mjx-math display="true" class="MJX-TEX" aria-hidden="true" style="margin-left: 0px; margin-right: 0px;"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.032em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c27E8"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mtext class="mjx-n" space="2"><mjx-c class="mjx-cA0"></mjx-c></mjx-mtext><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.182em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c27E9"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c27E8"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mtext class="mjx-n" space="2"><mjx-c class="mjx-cA0"></mjx-c></mjx-mtext><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-script></mjx-msub><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.104em;"><mjx-texatom size="s" texclass="ORD"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c27E9"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c27E8"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-mi class="mjx-i" size="s"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-script></mjx-msub><mjx-msubsup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.247em; margin-left: -0.104em;"><mjx-mi class="mjx-n" size="s" style="margin-left: 0.225em;"><mjx-c class="mjx-c22A4"></mjx-c></mjx-mi><mjx-spacer style="margin-top: 0.189em;"></mjx-spacer><mjx-texatom size="s" texclass="ORD"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msubsup><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mtext class="mjx-n" space="2"><mjx-c class="mjx-cA0"></mjx-c></mjx-mtext><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c27E9"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c27E8"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msubsup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.247em; margin-left: 0px;"><mjx-mo class="mjx-var" size="s"><mjx-c class="mjx-c2032"></mjx-c></mjx-mo><mjx-spacer style="margin-top: 0.248em;"></mjx-spacer><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msubsup><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mtext class="mjx-n" space="2"><mjx-c class="mjx-cA0"></mjx-c></mjx-mtext><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c27E9"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>d</mi><msub><mi>S</mi><mrow data-mjx-texclass="ORD"><mi>t</mi><mo>,</mo><mi>e</mi></mrow></msub><mo>=</mo><mo fence="false" stretchy="false">⟨</mo><mi>d</mi><msub><mi>O</mi><mi>t</mi></msub><mo>,</mo><mtext>&nbsp;</mtext><msub><mi>Y</mi><mrow data-mjx-texclass="ORD"><mi>e</mi><mo>,</mo><mi>t</mi></mrow></msub><mo fence="false" stretchy="false">⟩</mo><mo>=</mo><mo fence="false" stretchy="false">⟨</mo><mi>d</mi><msub><mi>O</mi><mi>t</mi></msub><mo>,</mo><mtext>&nbsp;</mtext><msub><mi>A</mi><mi>e</mi></msub><msub><mi>W</mi><mrow data-mjx-texclass="ORD"><mn>2</mn><mo>,</mo><mi>e</mi></mrow></msub><mo fence="false" stretchy="false">⟩</mo><mo>=</mo><mo fence="false" stretchy="false">⟨</mo><mi>d</mi><msub><mi>O</mi><mi>t</mi></msub><msubsup><mi>W</mi><mrow data-mjx-texclass="ORD"><mn>2</mn><mo>,</mo><mi>e</mi></mrow><mi mathvariant="normal">⊤</mi></msubsup><mo>,</mo><mtext>&nbsp;</mtext><msub><mi>A</mi><mrow data-mjx-texclass="ORD"><mi>e</mi><mo>,</mo><mi>t</mi></mrow></msub><mo fence="false" stretchy="false">⟩</mo><mo>=</mo><mo fence="false" stretchy="false">⟨</mo><mi>d</mi><msubsup><mi>A</mi><mrow data-mjx-texclass="ORD"><mi>e</mi><mo>,</mo><mi>t</mi></mrow><mo data-mjx-alternate="1">′</mo></msubsup><mo>,</mo><mtext>&nbsp;</mtext><msub><mi>A</mi><mrow data-mjx-texclass="ORD"><mi>e</mi><mo>,</mo><mi>t</mi></mrow></msub><mo fence="false" stretchy="false">⟩</mo></math></mjx-assistive-mml></mjx-container> <p>Neither <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="85" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msubsup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.247em; margin-left: 0px;"><mjx-mo class="mjx-var" size="s"><mjx-c class="mjx-c2032"></mjx-c></mjx-mo><mjx-spacer style="margin-top: 0.198em;"></mjx-spacer><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msubsup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><msubsup><mi>A</mi><mrow data-mjx-texclass="ORD"><mi>e</mi><mo>,</mo><mi>t</mi></mrow><mo data-mjx-alternate="1">′</mo></msubsup></math></mjx-assistive-mml></mjx-container> nor <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="86" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D452 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D461 TEX-I"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>A</mi><mrow data-mjx-texclass="ORD"><mi>e</mi><mo>,</mo><mi>t</mi></mrow></msub></math></mjx-assistive-mml></mjx-container> depends on <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="87" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>Y</mi></math></mjx-assistive-mml></mjx-container> or <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="88" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container>.</p>
<p class="zh-tr">Activation Memory 与 Expert Granularity 解耦</p> </li> </ul> </li> </ul> <h4 id="activation-memory-independent-of-expert-granularity">Activation Memory Independent of Expert Granularity</h4>
<h4 class="zh-h" id="activation-memory-independent-of-expert-granularity">SonicMoE 的 forward pass：只 cache $X$ 与 $H$。$X$ 的 gather 结果永不 cache 或 materialize；expert aggregation kernel 把 scatter 与 sum 融合。</h4> <p><strong>SonicMoE’s forward pass.</strong> In the forward pass, SonicMoE only caches <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="89" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="90" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>H</mi></math></mjx-assistive-mml></mjx-container>. The gathered results for <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="91" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> are <em>never</em> cached or materialized. The expert aggregation kernel fuses the scatter and summation together.</p>
<p class="zh-tr">图：SonicMoE 的 forward 计算 workflow，与 PyTorch 标准 MoE 实现对比；同时对比两种方法的 activation memory 与 IO cost。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/forward-workflow.png" width="100%"></p> <p align="center"><em>Figure: SonicMoE's forward computational workflow and comparison with a standard MoE implementation in PyTorch. We also compare the activation memory and IO costs for both methods.</em></p>
<p class="zh-tr" align="center">下图给出 activation memory 拆解的简要对比。SonicMoE 只 cache 输入 $X$ 与 pre-SwiGLU activation $H$，且不需要任何 GEMM recomputation。</p> <p>The following figure gives a brief comparison on the activation memory breakdown. SonicMoE caches only inputs <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="92" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> and pre-SwiGLU activation <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="93" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>H</mi></math></mjx-assistive-mml></mjx-container> and <em>does not need any GEMM recomputation</em>.</p>
<p class="zh-tr">图：用不同训练 kernel 时，Qwen3-235B MoE 模型单层（microbatch=32k）的 cached activation memory 示意。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/moe-activation-memory-qwen.png" width="40%"></p> <p align="center"><em>Figure: illustration of cached activation memory for a single layer of Qwen3-235B MoE model (microbatch=32k) when equipped with different training kernels.</em></p>
<p class="zh-tr" align="center">SonicMoE 在不增加任何训练 FLOPs 的前提下，能达到与同等 active 参数 dense 模型相同的 activation memory 效率。</p> <blockquote> <p>SonicMoE can achieve the same activation memory efficiency as a dense model with the same activated number of parameters without extra training FLOPs.</p>
<p class="zh-tr">通过算法重排降低 IO Cost</p> </blockquote> <h4 id="io-cost-reduction-through-algorithmic-reordering">IO Cost Reduction through Algorithmic Reordering</h4>
<h4 class="zh-h" id="io-cost-reduction-through-algorithmic-reordering">每少 cache 一个变量，就少一次 HBM 读或写。同一次「消灭 $O(TKd)$ activation」的重设计，也消灭了对应的 HBM round-trip。</h4> <p>Each variable that is no longer cached is also one fewer read or write to HBM. The same redesign that eliminates <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="94" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>O</mi><mo stretchy="false">(</mo><mi>T</mi><mi>K</mi><mi>d</mi><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container>-sized activations eliminates the corresponding HBM round-trips.</p>
<p class="zh-tr">SonicMoE 的 forward pass：把 gather 与 SwiGLU 融合进 up-projection；scatter $Y$ 与 expert aggregation 融合。</p> <p><strong>SonicMoE’s forward pass.</strong> We fuse the gather and SwiGLU activation in the up-projection. The scatter <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="95" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> operation is fused with the expert aggregation.</p>
<p class="zh-tr">SonicMoE 的 backward pass：</p> <p><strong>SonicMoE’s backward pass.</strong></p>
<p class="zh-tr">Activation gradient：down-proj 激活梯度 $dH$ kernel 同时计算 $dH$、$dS$、$A'$（$dW_2$ 的输入），全程不需要 cache $Y$ 或 $dY$。同样把 dSwiGLU 与 gather 融合进 GEMM。</p> <ul> <li> <p><strong>Activation gradient</strong>: The down-proj activation grad <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="96" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> kernel computes <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="97" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container>, <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="98" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>S</mi></math></mjx-assistive-mml></mjx-container>, and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="99" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: 0.363em;"><mjx-mo class="mjx-var" size="s"><mjx-c class="mjx-c2032"></mjx-c></mjx-mo></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>A</mi><mo data-mjx-alternate="1">′</mo></msup></math></mjx-assistive-mml></mjx-container> (input for <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="100" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.104em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><msub><mi>W</mi><mn>2</mn></msub></math></mjx-assistive-mml></mjx-container>) simultaneously, none of which require caching <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="101" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> or <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="102" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>Y</mi></math></mjx-assistive-mml></mjx-container>. We similarly fuse dSwiGLU and the gather operation into the GEMM.</p>
<p class="zh-tr">图：SonicMoE 的 backward activation gradient 计算 workflow，与 PyTorch 标准 MoE 实现对比。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/backward-activation-workflow.png" width="100%"></p> <p align="center"><em>Figure: SonicMoE's backward computational workflow for activation gradient and comparison with a standard MoE implementation in PyTorch.</em></p>
<p class="zh-tr" align="center">Weight gradient：$dW_1$ 与 $dW_2$ 的 weight gradient kernel 在执行时即时 gather $X$ 与 $dO$。算法层面 IO cost 与标准 MoE 一致，但 SonicMoE 的 gather fusion 通过利用 L2 cache locality 降低实际硬件 IO cost（稍后讨论）。</p> </li> <li> <p><strong>Weight gradient</strong>: The weight gradient kernels for <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="103" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.104em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c31"></mjx-c></mjx-mn></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><msub><mi>W</mi><mn>1</mn></msub></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="104" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.104em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><msub><mi>W</mi><mn>2</mn></msub></math></mjx-assistive-mml></mjx-container> gather <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="105" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="106" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container> on the fly during execution. While their <em>algorithmic IO costs</em> match a standard MoE implementation, SonicMoE’s gather fusion reduces the <em>hardware IO costs</em> by exploiting L2 cache locality, which we will discuss later.</p>
<p class="zh-tr">图：SonicMoE 的 backward weight gradient 计算 workflow，与 PyTorch 标准 MoE 实现对比。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/backward-weight-workflow.png" width="80%"></p> <p align="center"><em>Figure: SonicMoE's backward computational workflow for weight gradient and comparison with a standard MoE implementation in PyTorch.</em></p>
<p class="zh-tr" align="center">净效果：在任何硬件特定优化之前，IO cost 已经大幅降低：</p> </li> </ul> <p>The net effect is a large reduction in IO costs even before any hardware-specific optimizations:</p>
<p class="zh-tr">图：用不同训练 kernel 时，Qwen3-235B MoE 模型单层（microbatch=32k）的 IO cost 示意。SonicMoE 的 workflow 绕过了多次大型 tensor 的读写。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/moe-io-costs-qwen-fwd.png" width="40%"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/moe-io-costs-qwen-bwd.png" width="40%"></p> <p align="center"><em>Figure: Illustration of IO costs for a single layer of Qwen3-235B MoE model (microbatch=32k) when equipped with different training kernels. SonicMoE's workflow circumvents the need to read or write multiple massive-sized tensors compared to existing MoE kernels.</em></p>
<p class="zh-tr" align="center">在这些 kernel 中，特别强调反向 down-proj 激活梯度 $dH$ kernel —— 它结合了 IO-aware 与 hardware-aware 的算法设计：</p> <p>Among these kernels, we want to give a special highlight to our backward down-proj activation gradient <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="107" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> kernel as a combination of IO-aware and hardware-aware algorithmic design:</p>
<p class="zh-tr">图：SonicMoE 的 dH workflow 在语义上等价于 PyTorch 标准 MoE 实现的多个 kernel，但 SonicMoE 大幅降低 IO cost。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/dH-kernel-comparison.png" width="100%"></p> <p align="center"><em>Figure: the semantics of SonicMoE's dH workflow diagram is equivalent to standard PyTorch MoE implementation for multiple kernels while SonicMoE substantially reduces the IO costs. </em></p>
<p class="zh-tr" align="center">IO cost 削减：gather $dO$、fuse dSwiGLU 调用、不读不写 $Y$ 与 $dY$。</p> <ul> <li> <p><strong>reduction of IO costs</strong>: we gather <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="108" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container>, fuse the dSwiGLU call, and do not read or write <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="109" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="110" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44C TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>Y</mi></math></mjx-assistive-mml></mjx-container>.</p>
<p class="zh-tr">硬件异步特性进一步隐藏剩余 IO cost 延迟（稍后讨论）：dH kernel 设计已削减 IO cost，我们再用 modern NVIDIA GPU 的异步特性把剩余影响最小化。</p> </li> <li> <p><strong>hardware asynchrony features that further hide the remaining IO cost latency</strong> (will discuss later): the design of this <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="111" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> kernel already reduces IO costs, and we further minimize the remaining impact of IO costs by leveraging the asynchrony features on modern NVIDIA GPUs.</p>
<p class="zh-tr">图：可借助近期 NVIDIA 硬件特性把 SonicMoE dH kernel 的 IO 延迟藏起来，大幅降低整体 runtime。</p> </li> </ul> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/backward-dH-overlap.png" width="50%"></p> <p align="center"><em>Figure: we can leverage recent NVIDIA hardware features to hide the IO latency in SonicMoE's dH kernel and greatly reduce the overall runtime. </em></p>
<p class="zh-tr" align="center">精心的算法设计足以解决 activation memory 问题，并部分解决 IO cost 问题。我们可以借助硬件异步性进一步减小 IO cost 的影响。</p> <blockquote> <p>A careful algorithmic design is sufficient to address the activation memory issue and partially the IO cost issue. We can further minimize the impact of IO costs by leveraging hardware asynchrony.</p>
<p class="zh-tr">我们希望 SonicMoE 在 Hopper 和 Blackwell 上都达到峰值吞吐，所以对 SonicMoE 的所有 Grouped GEMM kernel 都应用 hardware-aware 优化。然而 modern NVIDIA GPU 架构在 execution model 上往往差异巨大。为此我们构建一个统一且模块化的软件抽象，把所有 Grouped GEMM kernel 表达为同一个结构，同时把架构特定优化局限到少量 override。本文余下部分描述这个抽象以及它在每个架构上的实现。</p> </blockquote> <p>We want SonicMoE to achieve peak throughput on both Hopper and Blackwell GPUs, so we apply hardware-aware optimizations to all Grouped GEMM kernels in SonicMoE. However, modern NVIDIA GPU architectures often differ substantially in their execution models. <strong>In response, we build a unified and modular software abstraction that expresses all grouped gemm kernels while localizing all architecture-specific optimizations to a small number of overrides.</strong> The rest of this post describes that abstraction and how it is realized on each architecture.</p>
<p class="zh-tr">2. 赋能 SonicMoE 的 QuACK 软件抽象</p> <h2 id="2-the-software-abstraction-of-quack-that-empowers-sonicmoe">2. the Software Abstraction of QuACK that Empowers SonicMoE</h2>
<h2 class="zh-h" id="2-the-software-abstraction-of-quack-that-empowers-sonicmoe">SonicMoE 已支持 NVIDIA Hopper（SM90）、Blackwell（SM100），SM120（Blackwell GeForce）支持也在路上。最初考虑把 Hopper kernel 移植到 Blackwell 时，最直接的路径是从头重写 6 个 Grouped GEMM kernel。我们最终选择抽出共享结构 —— 这一决定后来证明非常高产。</h2> <p>SonicMoE already supports NVIDIA Hopper (SM90), Blackwell GPUs (SM100), and the support for Blackwell GeForce (SM120) GPUs is on the way. When we first considered porting the Hopper kernels to Blackwell, the straightforward path was to rewrite 6 Grouped GEMM kernels from scratch. We chose instead to factor out the shared structure, and this decision proved highly productive later.</p>
<p class="zh-tr">每个 Grouped GEMM kernel 都是同一种底层结构的实例：一个 producer-consumer GEMM mainloop（让数据搬运与 tensor core 计算 overlap），跟一个参数化 epilogue（在数据落到 HBM 之前对 accumulator apply fusion 逻辑）。</p> <p>Every Grouped GEMM kernel is an instance of the same underlying structure: <strong>a producer-consumer GEMM mainloop that overlaps data movement with tensor core computation, followed by a parameterized epilogue</strong> that applies fusion logic directly to the accumulator before any data reaches HBM.</p>
<p class="zh-tr">这种 GEMM mainloop + customizable epilogue 的共享结构让 SonicMoE 实现模块化、可扩展到新硬件且仍能维持峰值性能。</p> <blockquote> <p>This shared structure of GEMM mainloop with customizable epilogue would make SonicMoE’s implementation modular, extendable to new hardware while still maintaining peak performance.</p>
<p class="zh-tr">我们也统一了 API 并封装其他架构特定改动。SonicMoE 的 GEMM kernel 建在 QuACK 之上 —— 我们自研的 CuTeDSL 库，重度借鉴 CUTLASS 与 CuTeDSL 官方 example。CUTLASS 为 GPU kernel 定义了一个干净的分层 programming model：mainloop 把 matrix multiplication 在并行 worker（Streaming Processor）上 tile 化，epilogue 在写回内存前后处理结果。QuACK 沿用这个分层 programming model，并加入 tile scheduler、customizable epilogue 等模块化组件。</p> </blockquote> <p>We also unify the API and encapsulate other architecture-specific changes. <strong>SonicMoE’s GEMM kernels are built on top of <a href="https://github.com/Dao-AILab/quack" rel="external nofollow noopener" target="_blank">QuACK</a>, our in-house CuTeDSL library that draws heavily from <a href="https://github.com/NVIDIA/cutlass" rel="external nofollow noopener" target="_blank">CUTLASS</a> and the <a href="https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL" rel="external nofollow noopener" target="_blank">CuTeDSL official examples</a>.</strong> CUTLASS defines a clean layered programming model for GPU kernels: a mainloop that tiles the matrix multiplication across the parallel workers (Streaming Processors), and an epilogue that post-processes the results before writing them back to memory. QuACK adopts this layered programming model and extends it with modular components (tile schedulers, customizable epilogue, etc.).</p>
<p class="zh-tr">下面我们看 QuACK GEMM 的设计、以及它如何帮助 SonicMoE 在高 IO cost 下达成峰值吞吐。</p> <p>Below, we examine the design of QuACK GEMM and how it helps SonicMoE achieve peak throughput amid high IO costs.</p>
<p class="zh-tr">NVIDIA GPU 上的 Tiled GEMM Kernel</p> <h3 id="tiled-gemm-kernel-on-nvidia-gpus">Tiled GEMM kernel on NVIDIA GPUs</h3>
<h3 class="zh-h" id="tiled-gemm-kernel-on-nvidia-gpus">NVIDIA GPU 上的 General Matrix Multiplication（GEMM）kernel 反复 fetch 输入 $A$、$B$ 的 tile（$A$ 通常是 activation，$B$ 通常是 weight），并把 tiled MMA（matrix multiply-accumulate）结果累加到一个零初始化的 buffer $C$（通常是 output activation）。</h3> <p>A General Matrix Multiplication (GEMM) kernel on NVIDIA GPUs repeatedly fetches tiles of input data <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="112" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="2"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi><mo>,</mo><mi>B</mi></math></mjx-assistive-mml></mjx-container> (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="113" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi></math></mjx-assistive-mml></mjx-container> is usually the activations while <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="114" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>B</mi></math></mjx-assistive-mml></mjx-container> is the weights), and we accumulate the tiled MMA (matrix multiply-accumulate) results into a zero-initialized buffer <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="115" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D436 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>C</mi></math></mjx-assistive-mml></mjx-container> (often the output activations).</p>
<p class="zh-tr">图：GEMM tiled accumulation 示意 [2]</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gemm.png" width="30%"></p> <p align="center"><em>Figure: illustration of GEMM tiled accumulation [2]</em></p>
<p class="zh-tr" align="center">每个 Output Tile 的三段累加</p> <h4 id="repeated-3-phase-accumulation-for-each-output-tile">Repeated 3-phase Accumulation for Each Output Tile</h4>
<h4 class="zh-h" id="repeated-3-phase-accumulation-for-each-output-tile">图：GPU 的每个 Streaming Processor (SM) 以 3 段方式执行 tiled MMA，直到所有 tile 处理完。通常会有一个 persistent tile scheduler 调度每个 SM 接收哪个 tile。改编自 [3]。</h4> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gemm-in-3-phase.png" width="60%"></p> <p align="center"><em>Figure: each Streaming Processor (SM) on GPUs will perform tiled MMA in 3 phases until no tiles left. Usually there will be a persistent tile scheduler that schedules which tile each SM will receive. Adapted from [3]. </em></p>
<p class="zh-tr" align="center">对每个 output tile，累加过程都被组织成三段：</p> <p>For every output tile, the accumulation process is formulated into three phases:</p>
<p class="zh-tr">Prologue（由 producer 完成）：load warp 把输入 load 进 SMEM buffer，填充 $A$、$B$ 的 tile。</p> <ul> <li> <p><strong>Prologue</strong> (by <em>producer</em>): the load warp(s) load the inputs to fill SMEM buffers with tiles of A and B.</p>
<p class="zh-tr">Mainloop（producer 负责 input load、consumer 负责 MMA）：MMA warp/warpgroup 消费已填好的 SMEM buffer，执行 MMA 指令，累加到 output buffer。Hopper 上结果 buffer 在 register（WGMMA）；Blackwell 上结果在 TMEM（UMMA）。</p> </li> <li> <p><strong>Mainloop</strong> (input loading by <em>producer</em>, MMA by <em>consumer</em>): the MMA warp/warpgroup consumes filled shared memory (SMEM) buffers, executes the MMA instruction, and accumulates into an output buffer. On Hopper this result buffer lives in registers (WGMMA). On Blackwell the result lives in TMEM (UMMA).</p>
<p class="zh-tr">Epilogue（由 consumer 完成）：consumer warpgroup（Hopper）或 dedicated epilogue warps（Blackwell）对累加结果 apply 任何 fused 后处理，并写回 GMEM（global memory，通常即 HBM）。</p> </li> <li> <p><strong>Epilogue</strong> (by <em>consumer</em>): the consumer warpgroup (Hopper) or the dedicated epilogue warps (Blackwell) apply any fused post-processing to the accumulated results, and write back to GMEM (global memory, often the HBM).</p>
<p class="zh-tr">这个三段结构对 MoE 中的 6 个 Grouped GEMM kernel 都一样。kernel 之间变化的仅有：</p> </li> </ul> <p>This three-stage structure is the same for all 6 Grouped GEMM kernels in MoE. What changes between kernels is exclusively the following:</p>
<p class="zh-tr">(1) 即 §4 描述的 gather fusion；(2) 即所有 MoE-specific fusion 逻辑的所在 —— QuACK customizable epilogue 抽象的核心。</p> <ol> <li>How the producer loads the data when we have contiguous or gathered inputs</li> <li>What the epilogue consumer does to the accumulator before writing it to GMEM</li> </ol> <p>Point (1) is the gather fusion described in Section 4. Point (2) is where all MoE-specific fusion logic lives, and it is the core of QuACK’s customizable epilogue abstraction.</p>
<p class="zh-tr">Tile Scheduling：决定每个 CTA 处理哪个 output tile</p> <h4 id="tile-scheduling-decide-which-output-tile-to-process-by-each-cta">Tile Scheduling: Decide which Output Tile to Process by Each CTA</h4>
<h4 class="zh-h" id="tile-scheduling-decide-which-output-tile-to-process-by-each-cta">persistent tile scheduler 把 unique tile coordinate 分给每个 CTA（thread block，通常每 SM 一个），直到所有 tile 消费完。根据架构与 kernel 配置自动选择多种 tile scheduler 模式：</h4> <p>A persistent tile scheduler will give a unique tile coordinate to each CTA (thread block, usually 1 per SM) until all tiles are consumed. Multiple modes of tile schedulers are supported and selected automatically based on architecture and kernel configuration:</p>
<p class="zh-tr">Static（SM90 默认）：固定的 linear tile-to-CTA 分配。</p> <ul> <li> <p><strong>Static</strong> (SM90 default): fixed linear tile-to-CTA assignment.</p>
<p class="zh-tr">Cluster Launch Control（CLC，SM100 默认）：通过 Blackwell 特有的 PTX 指令 <code>clusterlaunchcontrol.try_cancel</code> 实现的硬件辅助 cluster-level 动态调度。硬件管理 work queue。§3 详细描述。</p> </li> <li> <p><strong>Cluster Launch Control (CLC)</strong> (SM100 default): hardware-assisted cluster-level dynamic scheduling via the Blackwell-specific <code class="language-plaintext highlighter-rouge">clusterlaunchcontrol.try_cancel</code> PTX instruction. The hardware manages the work queue. We will describe CLC in detail in Section 3.</p>
<p class="zh-tr">Customizable Epilogue</p> </li> </ul> <h3 id="customizable-epilogue">Customizable Epilogue</h3>
<h3 class="zh-h" id="customizable-epilogue">base GEMM class 把 epilogue 实现为固定 loop skeleton。对每个 output sub-tile：</h3> <p>The base GEMM class implements the epilogue as a fixed loop skeleton. For each sub-tile of the output:</p>
<p class="zh-tr"><code>epi_visit_subtile</code> 在 base class 中是 no-op。Subclass override 它注入任意 per-element fusion。整个 SonicMoE 代码库里所有的 activation function、所有的 backward 计算、所有的 scaling、所有的 reduction，都从这一个方法注入。</p> <ol> <li>Load the accumulator fragment into a register tensor</li> <li>Call <code class="language-plaintext highlighter-rouge">epi_visit_subtile</code> to <strong>execute customized epilogue ops</strong>.</li> <li>Write epilogue results to shared memory and finally to global memory</li> </ol> <p>The <code class="language-plaintext highlighter-rouge">epi_visit_subtile</code> method is a no-op in the base class. Subclasses override it to inject arbitrary per-element fusion logic. <strong>This single method is the injection point for every activation function, every backward pass computation, every scaling operation, and every reduction in the entire SonicMoE codebase.</strong></p>
<p class="zh-tr">每个 epilogue mixin（如 SwiGLU 用的 <code>GemmGatedMixin</code>、$dH$ 反向用的 <code>GemmDGatedMixin</code>）配一个 architecture-specific base class：<code>GemmGatedSm90</code> / <code>GemmGatedSm100</code>、<code>GemmDGatedSm90</code> / <code>GemmDGatedSm100</code> 等。架构后缀只控制 warp layout、accumulator 移动（register vs. tensor memory）、硬件资源管理。<code>epi_visit_subtile</code> 中的 epilogue fusion 逻辑跨架构共享。例如 SonicMoE 最重的 kernel 就是带额外参数的 <code>GemmDGatedMixin</code>，仅 88 行：</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>"200 LoC + 88 LoC" 这数字背后的工程哲学</strong>
<p>这个抽象的关键不是"少写代码"，而是<b>把跨架构变化的边界画对了</b>。一个 fusion 写一遍跑两个架构，是因为：</p>
<ol>
<li>MMA 指令、累加器位置、scheduler 这些"硬件用法"的差异被压进 <code>GemmBaseSm90</code> / <code>GemmBaseSm100</code>；</li>
<li>fusion "算什么、写什么"的逻辑只依赖<em>累加器内容 + 几个外部 tensor</em>，与累加器物理位置无关 —— 写在 Mixin 里跨架构都对。</li>
</ol>
<p>工程上这是经典的 <b>SRP + template method</b> 模式落到 GPU kernel DSL。能这样做的前提是 CuTeDSL 把 GMEM/SMEM/TMEM/register 之间的 copy 抽象成统一的 <code>cute.copy(atom, src, dst)</code>，"换 atom"成为换硬件的接缝。</p>
<p>具体例子：dH kernel 里的 <code>colvec_reduce</code> —— Hopper 上累加器在 register，行归约走 warp shuffle；Blackwell 上累加器在 TMEM，需要先 <code>tcgen05.ld</code> 拉到 register 再做。这两条 path 看起来不同，但<b>从 epilogue 作者视角</b>都只是"我有一个 [BLK_M, I] 的 tile，在 I 维 reduce"。差异藏在 <code>GemmBaseSmXX</code> 的 sub-tile loader 里。</p>
</div> <p>Each epilogue mixin (e.g., <code class="language-plaintext highlighter-rouge">GemmGatedMixin</code> for SwiGLU, <code class="language-plaintext highlighter-rouge">GemmDGatedMixin</code> for the <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="116" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> backward) is paired with an architecture-specific base class: <code class="language-plaintext highlighter-rouge">GemmGatedSm90</code> / <code class="language-plaintext highlighter-rouge">GemmGatedSm100</code>, <code class="language-plaintext highlighter-rouge">GemmDGatedSm90</code> / <code class="language-plaintext highlighter-rouge">GemmDGatedSm100</code>, etc. The architecture-specific suffix controls only the warp layout, accumulator movement (registers vs. tensor memory), and hardware resource management. <strong>The epilogue fusion logic in <code class="language-plaintext highlighter-rouge">epi_visit_subtile</code> is shared across architectures.</strong> For example, the heaviest kernel in SonicMoE is just a <code class="language-plaintext highlighter-rouge">GemmDGatedMixin</code> with additional arguments, implemented in 88 lines:</p>
<p class="zh-tr">图：用 QuACK 实现的两个 SonicMoE kernel。左：kernel workflow；中：每个 kernel override <code>epi_visit_subtile</code> 的 QuACK epilogue mixin class（dH 88 LoC，up-proj forward 21 LoC）；右：SonicMoE 简化的 kernel 调用。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/quack-sonicmoe-code.png" width="100%"></p> <p align="center"><em>Figure: Two SonicMoE kernels implemented with QuACK. Left: the kernel workflow diagram. Center: the QuACK epilogue mixin class where each kernel overrides `epi_visit_subtile` (88 LoC for dH, 21 LoC for up-proj forward). Right: SonicMoE's simplified kernel launch call. </em></p>
<p class="zh-tr" align="center">总体上，QuACK 软件抽象交付我们看重的三项性质：</p> <p>In total, this software abstraction on QuACK delivers three properties we prioritize:</p>
<p class="zh-tr">对新模型架构 / 新算法的适配性：未来开发者只需修改 epilogue 就能为其他模型架构或算法（不只是 MoE）提供快速 kernel 实现。</p> <ul> <li> <p><strong>Adaptability to new model architecture or algorithms</strong>: future developers need only modify how epilogue works to provide a fast kernel implementation for other model architectures or algorithms, not only MoE.</p>
<p class="zh-tr">用这些抽象，我们能用 160 行同时为 Hopper 与 Blackwell 实现 symmetric GEMM kernel，并取得 SOTA 性能。</p> <ul> <li>For example, <a href="https://github.com/Dao-AILab/gram-newton-schulz" rel="external nofollow noopener" target="_blank">Gram Newton-Schulz</a> is also built on top of symmetric gemm on QuACK, with the quote from its blogpost: <blockquote> <p>Using these abstractions, we are able to implement the symmetric GEMM kernel for both Hopper and Blackwell in just 160 lines, while achieving SOTA performance.</p>
<p class="zh-tr">对新硬件（特性）的快速可扩展性：从顶到底跨硬件架构的统一 API。</p> </blockquote> </li> <li>We also only write <strong>~200 LoC</strong> to implement SonicMoE on top of QuACK Grouped GEMM which works automatically on both Hopper and Blackwell GPUs.</li> </ul> </li> <li> <p><strong>Fast extensibility to new hardware (features)</strong>: a unified API from top to bottom across different hardware architectures.</p>
<p class="zh-tr">改动 base GEMM 实现，已有 kernel 应该自动跑在新硬件上 —— 这让研究开发能快速迭代：</p> <p>We can change our base GEMM implementation and the existing kernels should work on the new hardware, which enables quick research development:</p>
<p class="zh-tr">我们把 TMA gather4 引入 Blackwell 的 Grouped GEMM，仅修改 copy atom 与 SMEM layout 约 100 行；MMA warp 完全不动。</p> <ul> <li> <p>We develop TMA gather4 for Grouped GEMM on Blackwell GPUs <a href="https://github.com/Dao-AILab/quack/commit/e282ee6529089d32d01fc178a1043b28bbf8bb9c#diff-fcdc3df7cf71ffdd7a3bde39db27fc4f729c71549614be61621441966393df2e" rel="external nofollow noopener" target="_blank">by simply modifying copy atoms and SMEM layouts</a> with ~100 LoC changes. <em>We do not change anything on the MMA warps.</em></p>
<p class="zh-tr">扩展 SM120（Blackwell GeForce GPU 如 5090）只需新增 base GEMM class 约 500 行；customizable epilogue 与 GEMM interface 完全不动。</p> </li> <li> <p>We extend to SM120 (Blackwell GeForce GPUs such as 5090) by simply adding <a href="https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm120.py" rel="external nofollow noopener" target="_blank">a base GEMM class</a> with ~500 LoC changes. <em>We do not change anything on the customizable epilogue and GEMM interface.</em></p>
<p class="zh-tr">代码库可维护性：新模块化设计降低未来维护成本，让代码库对新贡献者更友好。</p> </li> </ul> </li> <li> <p><strong>Codebase maintainability</strong>: the new modular design reduces the cost of future maintenance and makes the codebase accessible to new contributors.</p>
<p class="zh-tr">下一节描述 SonicMoE 如何受益于 Blackwell 的新特性。</p> <ul> <li>Our prior Hopper Grouped GEMM integrated 3-phase GEMM programming model and all possible fusions together, with more than 3k lines of code. This complexity placed a significant burden on maintainers and made adding new features error-prone.</li> </ul> </li> </ul> <p>In the next section, we will describe how SonicMoE benefits from new Blackwell features.</p>
<p class="zh-tr">3. 抽象底下：赋能 IO Overlap 的硬件特性</p> <h2 id="3-underneath-the-abstraction-hardware-features-that-empower-the-io-overlap">3. Underneath the Abstraction: Hardware Features that Empower the IO Overlap</h2>
<h2 class="zh-h" id="3-underneath-the-abstraction-hardware-features-that-empower-the-io-overlap">上一节的软件抽象之所以能把架构特定行为局限到少量 override，是因为 Blackwell 在硬件层提供了一些干净映射到这些 override 的新特性。本节描述这些硬件特性。</h2> <p>The software abstraction described in the previous section was designed so that all architecture-specific behavior is confined to a small number of localized overrides. This section describes what Blackwell provides at the hardware level, and why each new feature maps cleanly onto one of those overrides.</p>
<p class="zh-tr">GEMM Programming Model</p> <h3 id="gemm-programming-model">GEMM programming model</h3>
<h3 class="zh-h" id="gemm-programming-model">在 Hopper 上，MMA 通常用 warpgroup 级指令 WGMMA（<code>wgmma.mma_async</code>）执行：需要 128 个 thread（4 个连续 warp）一起 issue 与管理 —— warpgroup 内所有 thread 都参与跟踪 accumulator 状态，结果分布在 128 个 thread 的 register file 中。常用 2 个 consumer warpgroup，可以协同 issue 2 条 WGMMA，或让一个 warpgroup 的 IO 与另一个 warpgroup 的 GEMM 重叠。后者称为 "Ping-Pong warpgroup scheduling"，对带 heavy epilogue 的 kernel 特别有用 —— 一个 WG 做 MMA 时另一个跑 epilogue，互换角色。</h3> <p><strong>On Hopper</strong>, MMA is usually performed via a <em>warpgroup-level</em> instruction WGMMA (<code class="language-plaintext highlighter-rouge">wgmma.mma_async</code>). It requires 128 threads (4 contiguous warps) to issue and manage: all threads in the warpgroup participate in tracking the accumulator state, and the accumulator result is distributed across the register files of those 128 threads. We often have 2 consumer warpgroups, and we can either let them <em>cooperatively</em> issue 2 WGMMA instructions, or <strong>we can overlap the IO of 1 warpgroup with the GEMM of another warpgroup</strong>. In this case, we can let 1 consumer warpgroup do MMA while the other consumer warpgroup does the epilogue, and they switch roles once each finishes. This is called “Ping-Pong warpgroup scheduling”, often particularly useful to maintain high Tensor Core throughput with heavy epilogue.</p>
<p class="zh-tr">例如 down-proj forward kernel 的 epilogue 相对 mainloop 有较重的 HBM store IO；dH kernel 的 epilogue 需要 load $H$ 并执行多个 activation 与 reduction 操作来计算并存储 $dH$、$dS$、$A'$ 作为 $dW_2$ 的输入。</p> <p>For example, the down-proj forward kernel’s epilogue has heavy HBM store IO relative to the mainloop. In the <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="117" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> kernel’s epilogue, we need to load <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="118" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>H</mi></math></mjx-assistive-mml></mjx-container> and execute multiple activation and reduction operations to compute and store <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="119" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container>, <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="120" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D446 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>S</mi></math></mjx-assistive-mml></mjx-container>, and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="121" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msup><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: 0.363em;"><mjx-mo class="mjx-var" size="s"><mjx-c class="mjx-c2032"></mjx-c></mjx-mo></mjx-script></mjx-msup></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>A</mi><mo data-mjx-alternate="1">′</mo></msup></math></mjx-assistive-mml></mjx-container> as inputs for <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="122" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44A TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.104em;"><mjx-mn class="mjx-n" size="s"><mjx-c class="mjx-c32"></mjx-c></mjx-mn></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><msub><mi>W</mi><mn>2</mn></msub></math></mjx-assistive-mml></mjx-container>.</p>
<p class="zh-tr">图：Hopper Ping-Pong：两个 consumer warpgroup 在 MMA 与 epilogue 间交替。一个跑 Tensor Core MMA 时另一个跑 epilogue（TMA store + 任何 async load）。绿色箭头表示一个 warpgroup 给另一个的「可以继续」信号。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/pingpong-hopper.png" width="100%"></p> <p align="center"><em>Figure: Hopper Ping-Pong: two consumer warpgroups alternate between MMA and epilogue: while one runs Tensor Core MMA, the other runs the epilogue (TMA store + any async load). Green arrows show the signal from one warpgroup that the other can proceed. </em></p>
<p class="zh-tr" align="center">在 Blackwell 上，新的 UMMA（<code>tcgen05.mma</code>）指令彻底打破这种耦合。UMMA 是 single-threaded asynchronous 的：warp 内一个 thread issue 即可，执行异步进行，不占用其他 thread 或 register。accumulator 结果直接写到 Tensor Memory（TMEM）—— 每 SM 256 KB 的新型专用 on-chip 内存，与 register file 完全分离，物理上接到 tensor core。</p> <p><strong>On Blackwell</strong>, new UMMA (<code class="language-plaintext highlighter-rouge">tcgen05.mma</code>) instruction breaks this coupling entirely. UMMA is <em>single-threaded asynchronous</em>: one thread in the warp issues it, and execution proceeds asynchronously without occupying any other threads or registers. The accumulator result is written directly into Tensor Memory (TMEM) — a new dedicated 256 KB on-chip memory per SM that is wired into the tensor cores and completely separate from the register file.</p>
<p class="zh-tr">TMEM 物理布局：128 行 × 512 列 × 32-bit cell，共 256 KB / SM。512 列结构可容纳两个独立的 256 列 accumulator stage —— 这是 Blackwell MMA/epilogue overlap 的硬件基础。</p> <p>TMEM is organized as 128 rows × 512 columns of 32-bit cells, for a total of 256 KB per SM. The 512-column structure can hold two independent accumulator stages of 256 columns each. This is the hardware basis for Blackwell’s MMA/epilogue overlap as shown below.</p>
<p class="zh-tr">图：MMA warp 与 epilogue warp 之间的 TMEM 列所有权转移。这种技术常被称为 "double-buffering"。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/tmem-blackwell.png" width="50%"></p> <p align="center"><em>Figure: TMEM column ownership transfer between MMA warp and epilogue warps. This technique is often referred to as "double-buffering".</em></p>
<p class="zh-tr" align="center">MMA warp 累加到一个 256 列 stage 时，epilogue warps 同时通过 <code>tcgen05.ld</code>（TMEM-to-register copy 指令）drain 另一个 stage，并在之后跑 epilogue ops。epilogue warp 完成时通过 accumulator pipeline signal，MMA warp acquire 下一个 stage 开始填充。stage 在每个 tile 之间交替。这在精神上仍是 Ping-Pong —— overlap MMA 与 epilogue IO。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>TMEM 的几个不显然的硬件细节</strong>
<ul>
<li><b>TMEM 不是 SMEM 的扩展</b>：访问语义完全不同。TMEM 只能通过专用的 <code>tcgen05.ld / tcgen05.st</code> 在 TMEM 与 register 之间搬数据；MMA 直接消费 TMEM 中的累加器。普通 load/store 不能访问 TMEM。这意味着 epilogue 想用累加器内容必须先 <code>tcgen05.ld</code> 到 register 才能 store 到 GMEM 或与 register 中的其他张量做运算。</li>
<li><b>双 buffer 是手动管理的</b>：硬件提供两个 stage，但 pipeline 同步仍然要用 mbarrier。MMA 完成后 release 一个 stage 给 epilogue；epilogue 完成后 release 回 MMA。SonicMoE 用 QuACK 的 named barrier 抽象。</li>
<li><b>TMEM 容量限制 tile 大小</b>：256 KB / SM、双 buffer ⇒ 单 stage 128 KB。dH kernel 的累加器是 fp32 [BLK_M, I]，BLK_M=128、I=1536 时需要 768 KB —— 超出 → SonicMoE 把累加器按 N 维切，每个 sub-tile 走完 epilogue 再算下一个 sub-tile。</li>
<li><b>UMMA 的"单线程 issue"含义</b>：把发指令的线程释放出来不代表 warp 内其他线程闲着 —— 它们去做 producer / 调度 / 索引计算。这正是为什么 Blackwell 上"warp specialization"（producer warp / MMA warp / epilogue warp / scheduler warp 各司其职）成为标准范式。</li>
</ul>
</div> <p>While the MMA warp accumulates into one 256-column stage, the epilogue warps are simultaneously draining the other stage via <code class="language-plaintext highlighter-rouge">tcgen05.ld</code> (the TMEM-to-register copy instruction) and performing epilogue ops afterwards. When the epilogue warps finish and signal via the accumulator pipeline, the MMA warp acquires the next stage and begins filling it. The stages alternate every tile. <strong>This is Ping-Pong in spirit as it overlaps MMA with epilogue IO.</strong></p>
<p class="zh-tr">图：Blackwell warp-specialized pipeline：一个 producer warp（顶）、一个 MMA warp（中）、多个 epilogue warp（底）并发运行。绿色箭头表示 MMA 给 epilogue 的 TMEM stage ready 信号；黄色箭头表示 epilogue 给 MMA 的 TMEM stage release 信号。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/pingpong-blackwell.png" width="100%"></p> <p align="center"><em>Figure: Blackwell warp-specialized pipeline: one producer warp (top), one MMA warp (middle), multiple epilogue warps (bottom) running concurrently. Green arrows show the ready signal of TMEM stage from the MMA to epilogue warp. Yellow arrows show the release signal of TMEM stage from the epilogue to MMA warp. </em></p>
<p class="zh-tr" align="center">2CTA MMA</p> <h3 id="2cta-mma">2CTA MMA</h3>
<h3 class="zh-h" id="2cta-mma">Blackwell 的第二个主要特性是 UMMA 的 <code>cta_group::2</code> 变体。开启时，同 cluster 中的一对 CTA 协同执行单条 MMA 指令。tile 的 $M$ 维度翻倍：单 CTA UMMA 支持 $M_\text{tile}=128$，2CTA UMMA 支持 $M_\text{tile}=256$。</h3> <p>A second major Blackwell feature is the <code class="language-plaintext highlighter-rouge">cta_group::2</code> variant of UMMA. When this mode is enabled, a <em>pair</em> of CTAs in the same cluster cooperatively execute a single MMA instruction. The tile M dimension doubles: where a single-CTA UMMA supports up to <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="123" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c31"></mjx-c><mjx-c class="mjx-c32"></mjx-c><mjx-c class="mjx-c38"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><mo>=</mo><mn>128</mn></math></mjx-assistive-mml></mjx-container>, a 2CTA UMMA supports up to <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="124" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="4"><mjx-c class="mjx-c32"></mjx-c><mjx-c class="mjx-c35"></mjx-c><mjx-c class="mjx-c36"></mjx-c></mjx-mn></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><mo>=</mo><mn>256</mn></math></mjx-assistive-mml></mjx-container>.</p>
<p class="zh-tr">对形状 $M_\text{tile}\times N_\text{tile}\times K_\text{tile}$ 的 tile，FLOPs 为 $2M_\text{tile}N_\text{tile}K_\text{tile}$，从 SMEM load 的字节数为 $2(M_\text{tile}K_\text{tile} + N_\text{tile}K_\text{tile})$（$A$ 与 $B$）。固定 $N_\text{tile}$ 与 $K_\text{tile}$ 时，doubling $M_\text{tile}$ 让 FLOPs 翻倍但只多 $2M_\text{tile}K_\text{tile}$ 字节的 $A$ 数据 —— 形状 $N_\text{tile}\times K_\text{tile}$ 的 $B$ tile 在 CTA pair 间共享，所以每个 CTA 只 load 它独立做 2 个 1CTA tile 时所需 $B$ 数据的一半。这就是关键收益：$B$ tile 通过 TMA 在 CTA pair 间 multicast，每个 output 元素的 $B$ 侧 SMEM traffic 减半。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>为什么单 CTA + cluster 不行，必须 2CTA UMMA？</strong>
<p>Hopper 也有 cluster，cluster 内 CTA 也能用 TMA multicast 共享 tile。<b>但 Hopper 的 WGMMA 是 CTA-local 指令</b> —— 各 CTA 仍然各自累加自己的 $[128, N]$ 输出 tile，share B 不能让一条 MMA 指令覆盖更大的 $M$。</p>
<p>Blackwell 的 <code>cta_group::2</code> 变体让<em>一条 MMA 在硬件层面跨两个 CTA 协同</em>：leader CTA 看到的累加器扩展成 $[256, N]$，physical-wise 一半在 CTA0 的 TMEM、一半在 CTA1 的 TMEM；leader 的 issuing thread 触发 MMA 后，硬件让两个 SM 的 Tensor Core 同步消费这一份 B-tile。</p>
<p>工程上这要求 cluster size = 2，且需要 <b>cluster-scope barrier</b> 同步两个 CTA 的 SMEM 准备（详见后面的 relay warp）。SonicMoE 默认在 varlen-M 路径开 2CTA；varlen-K 路径有时不开（K 维度可能太短，2CTA 收益不明显）。</p>
</div> <p>For a tile of shape <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="125" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-msub space="3"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.085em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-msub space="3"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.04em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><mo>×</mo><msub><mi>N</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><mo>×</mo><msub><mi>K</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub></math></mjx-assistive-mml></mjx-container>, the number of FLOPs is <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="126" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.085em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.04em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mn>2</mn><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><msub><mi>N</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><msub><mi>K</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub></math></mjx-assistive-mml></mjx-container> and the number of bytes loaded from SMEM is <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="127" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.04em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-c2B"></mjx-c></mjx-mo><mjx-msub space="3"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.085em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.04em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mn>2</mn><mo stretchy="false">(</mo><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><msub><mi>K</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><mo>+</mo><msub><mi>N</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><msub><mi>K</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container> for A and B. For fixed <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="128" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.085em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>N</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="129" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.04em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>K</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub></math></mjx-assistive-mml></mjx-container>, doubling <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="130" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub></math></mjx-assistive-mml></mjx-container> doubles the FLOPs but only adds <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="131" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mn class="mjx-n"><mjx-c class="mjx-c32"></mjx-c></mjx-mn><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D440 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.081em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.04em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mn>2</mn><msub><mi>M</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><msub><mi>K</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub></math></mjx-assistive-mml></mjx-container> bytes of A data — the B tile of shape <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="132" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-msub><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D441 TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.085em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-msub space="3"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-script style="vertical-align: -0.15em; margin-left: -0.04em;"><mjx-texatom size="s" texclass="ORD"><mjx-mi class="mjx-n"><mjx-c class="mjx-c74"></mjx-c><mjx-c class="mjx-c69"></mjx-c><mjx-c class="mjx-c6C"></mjx-c><mjx-c class="mjx-c65"></mjx-c></mjx-mi></mjx-texatom></mjx-script></mjx-msub></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>N</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub><mo>×</mo><msub><mi>K</mi><mrow data-mjx-texclass="ORD"><mi data-mjx-auto-op="false">tile</mi></mrow></msub></math></mjx-assistive-mml></mjx-container> is <em>shared</em> across the pair, so each CTA loads only half the B data it would need for two independent 1CTA tiles. This is the key benefit: the B tile is multicasted via TMA across the CTA pair, halving B-side SMEM traffic per output element.</p>
<p class="zh-tr">图：独立 1CTA MMA（左）vs. 2CTA MMA（右，图中称为 2xSM MMA）。左：两个独立 CTA 各 load 完整的 $B$ tile 并在 TMEM 中各持完整 accumulator。右：2CTA MMA 中 $B$ tile 减半共享，每个 CTA 在 TMEM 中持完整 accumulator 但只 load 一半 $B$ 数据。[4]</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/2cta-mma.png" width="60%"></p> <p align="center"><em>Figure: independent 1CTA MMA (left) vs. 2CTA MMA, referred to as 2xSM MMA in the figure (right). Left: two separate CTAs each load a full B tile and hold a full accumulator in TMEM. Right: in 2CTA MMA, B tile is halved and shared. Each CTA holds the full accumulator on TMEM but loads only half the B data. [4] </em></p>
<p class="zh-tr" align="center">Native Dynamic Persistent Tile Scheduler</p> <h3 id="native-dynamic-persistent-tile-scheduler">Native Dynamic Persistent Tile Scheduler</h3>
<h3 class="zh-h" id="native-dynamic-persistent-tile-scheduler">persistent tile scheduler 对 MoE kernel 必不可少 —— 它允许一个 CTA 在当前 tile 的 epilogue 还在跑时就开始 load 下一个 tile，让 producer 与 consumer pipeline 持续忙碌。</h3> <p>A persistent tile scheduler is essential for MoE kernels because it allows one CTA to begin loading the next tile while the current tile’s epilogue is still in progress, keeping both the producer and consumer pipelines continuously occupied.</p>
<p class="zh-tr">Hopper 上常用固定的 linear tile-to-CTA 静态预分配（「static tile scheduler」）—— 零同步开销，但 expert token 数变化时易出现 workload 不均。要做 SM 进度感知的 dynamic persistent tile scheduler 就得用 GMEM 全局 semaphore counter 与 atomic traffic；dynamic 相对 static 的优势在 Hopper 上往往不明显或不决定性。</p> <p>On Hopper, we often have a fixed, <em>static</em> linear pre-assignment of tiles to CTAs (we call it “static tile scheduler”). This induces <em>zero synchronization overhead</em>, but it is susceptible to workload imbalance when expert token counts vary. Implementing a dynamic persistent tile scheduler aware of each SM’s progress requires a global semaphore counter in GMEM and atomic traffic. The advantage of dynamic persistent over static persistent is often not obvious or decisive.</p>
<p class="zh-tr">Blackwell 引入 Cluster Launch Control（CLC）：硬件指令 <code>clusterlaunchcontrol.try_cancel</code> 让运行中的 cluster 向硬件 query 下一个 tile 坐标，无需碰 GMEM atomics。硬件管理 work queue，按 cluster 粒度操作，返回 tile 坐标或所有 tile 处理完的 decline 信号。query 开销极小，response 一次广播给整个 cluster，完全消除 per-CTA atomic traffic。</p> <p>Blackwell introduces <strong>Cluster Launch Control (CLC)</strong>: a hardware instruction <code class="language-plaintext highlighter-rouge">clusterlaunchcontrol.try_cancel</code> that lets a running cluster query the hardware for its next tile coordinate without touching GMEM atomics. The hardware manages the work queue, operates at cluster granularity, and returns either a tile coordinate or a decline signal when all tiles are processed. The query to the hardware has minimal overhead and the response is broadcast to the whole cluster at once, eliminating per-CTA atomic traffic entirely.</p>
<p class="zh-tr">图：无 persistent tile scheduler（左）与有 CLC tile scheduler（右）的 SM heatmap [5]。CLC tile scheduler 让所有 SM 在 kernel runtime 期间保持 active。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/non-persistent-heatmap.png" width="40%"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/clc-heatmap.png" width="40%"></p> <p align="center"><em>Figure: SM heatmap without persistent tile scheduler (left) and with CLC tile scheduler (right) [5]. The CLC tile scheduler can help all SMs stay active throughout the kernel runtime. </em></p>
<p class="zh-tr" align="center">CLC tile scheduler 与 varlen-M Grouped GEMM 中 2CTA MMA 的广泛使用，已经让 SonicMoE 比 DeepGEMM <code>sm100_m_grouped_bf16_gemm_contiguous</code> 与 Triton 官方 example 都高约 10% 吞吐。我们在附录中给出 SonicMoE 实现与 DeepGEMM、Triton 官方 example 的对比。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>CLC 在 MoE 长尾下的微观行为</strong>
<p>假设 128 个 expert，每个收到的 token 数从 100 到 5000 不等。静态 linear scheduler 把 tile 按 (expert_id, m_tile_id) 排序后均分给所有 SM —— 收到 5000 token 的那个 expert 对应 ~40 个 tile，分到这 40 个 tile 的 SM 会 lag 几百 µs；其他 SM 跑完手头的 tile 就只能 idle 等。</p>
<p>CLC 下：每个 cluster 处理完一个 tile 后立刻 <code>try_cancel</code> 拿下一个；硬件 work queue 按 FIFO 出 tile，谁先完成谁先拿。长 expert 的尾巴被切成小块分散到全 grid。</p>
<p>代码对应：<code>sonicmoe/functional/tile_scheduler.py</code> 的 <code>SonicMoEVarlenMTileScheduler</code> 扩展 QuACK 的 <code>VarlenMTileScheduler</code>，加了 prefetch（提前 issue 下一个 try_cancel），把 query latency 也藏起来。</p>
<p>与 DeepGEMM 对比的 ~10% 优势拆分：CLC 约 3-5%，2CTA shared-B 约 5-7%。SonicMoE 比 Triton 官方 example 强的部分还包括 SMEM swizzle 与 warp layout 的微调。</p>
</div> <p><strong>The CLC tile scheduler and extensive use of 2CTA MMA in varlen-M Grouped GEMM already help SonicMoE to achieve higher throughput (~10\%) than both <a href="https://github.com/deepseek-ai/DeepGEMM/blob/d30fc36c8f229f4f873b90a492f6e19e6e610923/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp#L124" rel="external nofollow noopener" target="_blank">DeepGEMM sm100_m_grouped_bf16_gemm_contiguous</a> and <a href="https://github.com/triton-lang/triton/blob/7d0756121cc95d6971112fc5c1fa99107b892444/python/triton_kernels/triton_kernels/matmul_details/_p_matmul.py#L57" rel="external nofollow noopener" target="_blank">triton official example</a>.</strong> We compare SonicMoE’s implementation with the DeepGEMM and triton official example in the appendix.</p>
<p class="zh-tr">图：B300 GPU 上以连续打包输入跑的 varlen-M Grouped GEMM。其他 baseline 详细描述见 arXiv 论文 Figure 18 caption。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/grouped_gemm_benchmark-B300.png" width="100%"></p> <p align="center"><em>Figure: Varlen-M Grouped GEMM with contiguously-packed inputs on B300 GPUs. Detailed descriptions of other baselines can be found in the caption of Figure 18 of our arXiv paper. </em></p>
<p class="zh-tr" align="center">4. 减小 IO Cost 的影响</p> <h2 id="4-reducing-the-impact-of-io-costs">4. Reducing the Impact of IO Costs</h2>
<h2 class="zh-h" id="4-reducing-the-impact-of-io-costs">§3 描述的硬件特性提供高吞吐基础设施。但对 fine-grained MoE，主导成本不是裸 MMA throughput —— 而是从任意位置 gather token 的 IO 开销，以及在不让 tensor core stall 的前提下执行 heavy epilogue 计算的开销。本节描述应对这些成本的三个 fusion 原则、以及它们在 Blackwell 上的适配。</h2> <p>The hardware features described in Section 3 provide the infrastructure for high throughput. But for fine-grained MoE, the dominant cost is not raw MMA throughput: it is the IO overhead of gathering tokens from arbitrary positions and of executing heavy epilogue computations without stalling the tensor cores. This section describes the three fusion principles that address these costs, and how each one is adapted for Blackwell.</p>
<p class="zh-tr">Gather Fusion</p> <h3 id="gather-fusion">Gather Fusion</h3>
<h3 class="zh-h" id="gather-fusion">SonicMoE 中多个 varlen-M GEMM 从输入 tensor 的任意位置读 token —— routing 决定 $X$（或 $dO$）的哪些 row 属于哪个 expert。SonicMoE 把 gather 直接 fuse 进 GMEM-to-SMEM 的 load。Blackwell 上根据 autotuning 阶段的速度，dispatch 到 <code>cp.async</code> 或 TMA gather4（<code>cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4</code>，每条指令搬 4 行）。</h3> <p>Multiple varlen-M GEMMs in SonicMoE read tokens from arbitrary positions in the input tensor where the routing decision determines which rows of <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="133" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> (or <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="134" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container>) belong to each expert. SonicMoE fuses the gather directly into the GMEM-to-SMEM load. On Blackwell GPUs, SonicMoE will dispatch to gather with either <code class="language-plaintext highlighter-rouge">cp.async</code> or TMA gather4 (<code class="language-plaintext highlighter-rouge">cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4</code> gathers 4 rows each time), whichever is faster at autotuning stage.</p>
<p class="zh-tr">图：2CTA MMA relay 机制。CTA 0（顶）作为 leader CTA：1 个 warp fetch index、4 个 warp issue <code>cp.async</code> gather、1 个 warp 在 barrier 上等之后 issue 2CTA MMA 指令。CTA 1（底）：1 个 warp fetch index、4 个 warp issue <code>cp.async</code> gather、1 个 relay warp 等本 CTA 的 <code>cp.async</code> 完成后 arrive 到 CTA 0 的 barrier。</p> <ul> <li> <strong><code class="language-plaintext highlighter-rouge">cp.async</code> gather fusion with 2CTA MMA.</strong> When 2CTA MMA is combined with cp.async gather fusion, a synchronization challenge arises: cp.async can only signal completion within its own CTA, <strong>but the leader CTA’s MMA needs both CTAs’ data ready.</strong> We resolve this with a dedicated relay warp in CTA 1 (non-leader) that forwards the completion signal to CTA 0 (leader) via a cluster-scope barrier.</li> </ul> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/relay-2CTA.png" width="40%"></p> <p align="center"><em>Figure: 2CTA MMA relay mechanism. CTA 0 (top) as the leader CTA: 1 warp fetches indices, 4 warps issue `cp.async` gathers, 1 warp issues the 2CTA MMA instruction after waiting at its barrier. CTA 1 (bottom): 1 warp fetches indices, 4 warps issue `cp.async` gathers, 1 relay warp waits for the `cp.async` completion and then arrives at CTA 0's barrier. </em></p>
<p class="zh-tr" align="center">我们对比 SonicMoE 的 gather fusion 与其他 MoE kernel「独立 gather kernel 的 GEMM」或「带 gather fusion 的 GEMM」的速度。SonicMoE 的 gather fusion 相对 contiguous 输入，$M$ 维仅慢 1.4%、$K$ 维反而快 0.5%。因此 SonicMoE 即便带 gather fusion 仍然比 ScatterMoE、MoMoE、Triton 官方 example 持续高 TFLOPS。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>cp.async 与 TMA 在异步语义上的实质差异</strong>
<p><code>cp.async</code>（SM80 引入）的完成事件靠 <code>cp.async.commit_group</code> + <code>cp.async.wait_group</code>，<b>这两个语义都是 CTA-local 的</b> —— 同一 CTA 的 thread 用 <code>commit_group</code> 提交一组 inflight cp.async，用 <code>wait_group</code> 等到留下指定个数。</p>
<p>TMA（SM90 引入的 <code>cp.async.bulk.tensor</code> 系列）则把完成事件挂在 <b>mbarrier</b> 上 —— 这是 cluster-scope 可见的同步原语，cluster 内任何 CTA 都能等 TMA 完成。Blackwell 把 TMA 进一步扩展出 gather4、scatter4 等变体。</p>
<p>所以"用 TMA gather4 不需要 relay warp"是因为 TMA 自带 cluster-aware 完成事件；而 cp.async 走的是 SM80 时代的 CTA-local 完成事件，必须人工搭桥。</p>
<p>工程上保留两条路径是为了 autotuning 灵活：某些 shape 下 cp.async 的 issue rate 反而更高（不需要 descriptor setup），所以 SonicMoE 把"gather 用哪条"作为可调参数（实测 &lt; 2% 差异）。</p>
<p style="background:#fde7e7;border-left:4px solid #c9302c;padding:8px 12px;margin:8px 0;"><b>⚠ 陷阱：</b>relay warp 不能复用 producer warp（producer 自己也在发 cp.async，wait 自己的完成事件会 deadlock）。必须独立 1 个 warp 专做 relay。早期版本因为想省 warp 把 relay 合并到 producer，在 cluster=2 时出现间歇性 hang。</p>
</div> <p>We then compare the speed of SonicMoE’s gather fusion against other MoE kernels’ GEMM with a separate gather kernel or with gather fusion. SonicMoE’s gather fusion is only 1.4% slower on the M dimension and 0.5% faster on the K dimension relative to contiguous inputs. Therefore, SonicMoE consistently achieves higher TFLOPS than ScatterMoE, MoMoE, and the triton official example even with gather fusion.</p>
<p class="zh-tr">图：B300 GPU 上 forward up-proj（$M$ 维 gather）与 backward dW1 kernel（$K$ 维 gather）。SonicMoE 同时支持从不同位置 gather 的输入（不透明柱）与连续打包的输入（透明柱）。其他 baseline 描述见 arXiv 论文 Figure 19 caption。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gather_grouped_gemm_benchmark-B300.png" width="100%"></p> <p align="center"><em>Figure: Forward pass up-proj (gather on M dim) and backward dW1 kernel (gather on K dim) kernel on B300 GPUs. SonicMoE supports both inputs gathered from different positions (opaque bars) and contiguously-packed inputs (transparent bars). Detailed descriptions of other baselines can be found in the caption of Figure 19 of our arXiv paper. </em></p>
<p class="zh-tr" align="center">Gather Fusion 通过 L2 Cache Locality 降低硬件 IO Cost</p> <h4 id="gather-fusion-reduces-hardware-io-costs-via-l2-cache-locality">Gather Fusion Reduces <em>Hardware</em> IO costs via L2 Cache Locality</h4>
<h4 class="zh-h" id="gather-fusion-reduces-hardware-io-costs-via-l2-cache-locality">L2 cache 在 GPU memory 层级中位于 HBM 与 SMEM 之间，所有 SM 共享。SM↔HBM 的所有 traffic 都过 L2：命中时按 L2 带宽（~20 TB/s [7]）服务，不碰 HBM；miss 时从 HBM（7.7 TB/s）取回并写入 L2 供未来复用。</h4> <p>The L2 cache sits between HBM and SMEM in the GPU memory hierarchy and is shared across all SMs. All traffic between SMs and HBM flows through L2: when an SM requests data that is already cached, the request is served at L2 bandwidth (~20 TB/s [7]) without touching HBM. When the request misses, the data is fetched from HBM (7.7 TB/s) and inserted into L2 for future reuse.</p>
<p class="zh-tr">gather fusion 的常见替代方案是跑独立 gather kernel 把输入预先排成 contiguous buffer 再喂 Grouped GEMM。两种方法的算法 IO cost 相同（不考虑 $N$ 维 TMA multicast），但 gather fusion 通过更好的 L2 cache 利用降低实际 HBM load traffic。</p> <p>A common alternative to gather fusion is to run a separate gather kernel that pre-arranges the inputs into a contiguous buffer before the Grouped GEMM. Although both approaches have identical <em>algorithmic IO costs</em> (assuming no TMA multicast along the N dimension), gather fusion reduces the actual HBM load traffic through better L2 cache utilization.</p>
<p class="zh-tr">图：gather fusion（左）从 compact 的 source tensor 读。Contiguous load（右）从 $K$ 倍大的 tensor 读 —— 每个 token 在 $K$ 个不同地址被复制，working set 随 granularity 增大超过 L2 capacity。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gather-fusion-L2.png" width="100%"></p> <p align="center"><em>Figure: Gather fusion (left) reads from a compact source tensor. Contiguous load (right) reads from a K times larger tensor where each token is duplicated across K distinct addresses, expanding the working set beyond L2 capacity as granularity increases. </em></p>
<p class="zh-tr" align="center">尽管 gather fusion 与从 pre-gathered 输入 contiguous load 的算法 IO cost 相同，gather fusion 通过更高的 L2 cache hit rate 实现更低的硬件 HBM IO cost。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>反直觉的"算法 IO 一致 ≠ 硬件 IO 一致"</strong>
<p>很多工程师会想："不就是把 X 重排一下吗，HBM 流量怎么会变？" 关键洞察：<b>预先 gather 出来的 X_g 是 $T \times K \times d$，比原始 X 大 K 倍</b>。K=8、$Td = 256$ MB 时 X_g = 2 GB。B300 的 L2 是 192 MB，X_g 远超 L2 ⇒ 后续 GEMM 读 X_g 几乎全 miss。原始 X 只有 256 MB（仍超 L2 但少很多），且 GEMM 读 X 的访问 pattern 因为 gather indices 已按 expert 排过序，同一行可能被不同 tile 的 producer 重复读 → 反而能复用 L2。</p>
<p>实测（appendix）：$(T,d,n,E,K)=(32768, 2048, 512, 256, 32)$ 的 up-proj forward：</p>
<table style="border-collapse:collapse;font-size:13px;margin:8px 0;">
<thead><tr><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">路径</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">HBM load</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">L2 hit rate</th></tr></thead>
<tbody>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">Gather fusion (cp.async)</td><td style="padding:4px 10px;border:1px solid #ccc;"><b>2.20 GB</b></td><td style="padding:4px 10px;border:1px solid #ccc;"><b>74.9%</b></td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">Contiguous TMA on pre-gathered</td><td style="padding:4px 10px;border:1px solid #ccc;">2.68 GB</td><td style="padding:4px 10px;border:1px solid #ccc;">66.3%</td></tr>
</tbody></table>
<p>经验法则：<em>"宁可让访问稍微随机（在 SMEM 内 gather），也不要让中间张量超过 L2 capacity（B300 192 MB, H100 60 MB）"</em>。这对未来 MoE 系统设计普适 —— 比如 EP 下 token 重排时也应该尽量延迟到 GEMM 内做。</p>
</div> <blockquote> <p>Although gather fusion has the same <em>algorithmic IO costs</em> as contiguous load from pre-gathered inputs, <strong>gather fusion achieves lower hardware HBM IO costs via better L2 cache hit rate.</strong></p>
<p class="zh-tr">我们用 NCU profiling 验证这一点，详细结果见附录。</p> </blockquote> <p>We validate this with NCU profiling and present detailed results in the appendix.</p>
<p class="zh-tr">SwiGLU 与 dSwiGLU Fusion</p> <h3 id="swiglu-and-dswiglu-fusion">SwiGLU and dSwiGLU Fusion</h3>
<h3 class="zh-h" id="swiglu-and-dswiglu-fusion">SonicMoE 在数据离开 epilogue 之前就 in-register apply activation。GEMM accumulator 在 register 中持有 MMA 结果 sub-tile；SwiGLU 以 element-wise interleaved 格式 apply 产生 activation sub-tile。MMA 结果（$H$）与 SwiGLU activation（$A$）都通过 async TMA store 机制写到 HBM —— 不增加 critical path 延迟。</h3>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>SwiGLU 的 interleaved layout 与 dSwiGLU jacobian</strong>
<p>SwiGLU 写成 $\mathrm{SwiGLU}(h) = \mathrm{silu}(h_\text{gate}) \odot h_\text{up}$。up-proj 输出 $h$ 是 $[TK, 2I]$，前 $I$ 列是 $h_\text{gate}$、后 $I$ 列是 $h_\text{up}$。SonicMoE 默认 <b>interleaved layout</b>：$[\text{gate}_0, \text{up}_0, \text{gate}_1, \text{up}_1, ...]$，方便 register 内同时拿到 gate 与 up 一对元素做 fusion。可以通过 <code>concat_layout=True</code> 切换到 concat layout（前后两半）兼容某些 checkpoint。</p>
<p>dSwiGLU jacobian（dH kernel epilogue 用）：</p>
<p>$\dfrac{\partial \mathrm{SwiGLU}}{\partial h_\text{gate}} = \sigma(h_\text{gate})\big(1 + h_\text{gate}(1 - \sigma(h_\text{gate}))\big) \cdot h_\text{up}$</p>
<p>$\dfrac{\partial \mathrm{SwiGLU}}{\partial h_\text{up}} = \mathrm{silu}(h_\text{gate})$</p>
<p>两条都只需要 $h$ 一个张量 → 这就是为什么 dH kernel 反向只 cache <code>h</code>。</p>
<p>实现细节：<code>silu(x) = x · sigmoid(x)</code>，sigmoid 在 register 用 fast-math 近似，dSwiGLU jacobian 的 $\sigma(h_\text{gate})$ 计算一次后两式共用。所有这些都在一个 epilogue tile 内完成、不写 HBM。</p>
</div> <p>SonicMoE applies the activation function in-register before any data leaves the epilogue. The GEMM accumulator holds MMA result sub-tiles in registers. SwiGLU is applied element-wise in an interleaved format to produce activation sub-tiles. Both MMA results (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="135" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>H</mi></math></mjx-assistive-mml></mjx-container>) and SwiGLU activations (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="136" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi></math></mjx-assistive-mml></mjx-container>) will be written to the HBM via the async TMA store mechanism which does not add latency to the critical path.</p>
<p class="zh-tr">Overlapping IO with MMA Compute：dH Kernel</p> <h3 id="overlapping-io-with-mma-compute-dh-kernel">Overlapping IO with MMA Compute: <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="137" style="font-size: 119.5%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> kernel</h3>
<h3 class="zh-h" id="overlapping-io-with-mma-compute-dh-kernel">SonicMoE 在所有可能处都让 IO 与 MMA overlap。这里聚焦 dH kernel，它是 SonicMoE 中 epilogue 最重的 kernel。做法是通过拆分 TMEM 资源 + 专用 TMA pipeline，把 epilogue warp 与 MMA warp 的角色 overlap。</h3> <p>SonicMoE overlaps IO with MMA whenever possible. Here we focus on the <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="138" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> kernel which has the heaviest epilogue in SonicMoE. To address this, we overlap the role of epilogue warps with the role of MMA warp by splitting the TMEM resources and employing dedicated TMA pipeline.</p>
<p class="zh-tr">图：SonicMoE dH kernel 中 epilogue ops 与 GEMM MMA overlap 的示意。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/dH-kernel.png" width="50%"></p> <p align="center"><em>Figure: illustration of epilogue ops overlapped with GEMM MMA in SonicMoE's dH kernel. </em></p>
<p class="zh-tr" align="center">下图考察 SonicMoE dH kernel（带 heavy epilogue，左列）与 GEMM-with-normal-epilogue-store（右列）在 Qwen3-235B-A22B-Thinking-2507（$(T,d,n,E,K)=(32768,4096,1536,128,8)$）上的硬件单元利用率。MMA throughput 的下降亚比例于 epilogue IO cost 的上升：dH kernel epilogue 让 HBM traffic 多 24%（6.33 → 7.86 GB），但 Tensor Core 与 Tensor Memory 利用率仅从 98% 降到 88%，相应 TFLOPS 从 1213 降到 1078（下降 11%）。</p> <p>In the following figure, we examine the hardware unit utilization of SonicMoE’s <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="139" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> kernel with heavy epilogue (left column) or GEMM with normal epilogue store (right column) on Qwen3-235B-A22B-Thinking-2507 (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="140" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="2"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="2"><mjx-c class="mjx-c1D45B TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="2"><mjx-c class="mjx-c1D438 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="2"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c3D"></mjx-c></mjx-mo><mjx-mo class="mjx-n" space="4"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mn class="mjx-n"><mjx-c class="mjx-c33"></mjx-c><mjx-c class="mjx-c32"></mjx-c><mjx-c class="mjx-c37"></mjx-c><mjx-c class="mjx-c36"></mjx-c><mjx-c class="mjx-c38"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="2"><mjx-c class="mjx-c34"></mjx-c><mjx-c class="mjx-c30"></mjx-c><mjx-c class="mjx-c39"></mjx-c><mjx-c class="mjx-c36"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="2"><mjx-c class="mjx-c31"></mjx-c><mjx-c class="mjx-c35"></mjx-c><mjx-c class="mjx-c33"></mjx-c><mjx-c class="mjx-c36"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="2"><mjx-c class="mjx-c31"></mjx-c><mjx-c class="mjx-c32"></mjx-c><mjx-c class="mjx-c38"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c2C"></mjx-c></mjx-mo><mjx-mn class="mjx-n" space="2"><mjx-c class="mjx-c38"></mjx-c></mjx-mn><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo stretchy="false">(</mo><mi>T</mi><mo>,</mo><mi>d</mi><mo>,</mo><mi>n</mi><mo>,</mo><mi>E</mi><mo>,</mo><mi>K</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>32768</mn><mo>,</mo><mn>4096</mn><mo>,</mo><mn>1536</mn><mo>,</mo><mn>128</mn><mo>,</mo><mn>8</mn><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container>). <strong>The drop in MMA throughput is <em>subproportional</em> to the increase in epilogue IO costs:</strong></p>
<p class="zh-tr">图：B300 GPU 上 Qwen3-235B-A22B-Thinking-2507（microbatch=32k）下，SonicMoE dH kernel（带 4 个 epilogue ops，左列）与 Grouped GEMM alone（右列）的 Nsight Compute Profiling。上行：kernel runtime 内 Tensor Pipe（MMA）与 DRAM 实现的吞吐；下行：硬件单元上传输的字节数。</p> <ul> <li>The <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="141" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>H</mi></math></mjx-assistive-mml></mjx-container> kernel epilogue increases HBM traffic by 24% (6.33 to 7.86 GB).</li> <li>However, both the Tensor Core and Tensor Memory utilization only drop from 98% to 88% with the corresponding TFLOPS drop from 1213 to 1078 (11% decrease).</li> </ul> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/32k-4k-1.5k-128-8-dH.png" width="47%">&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/32k-4k-1.5k-128-8-gemm-alone.png" width="47.5%"></p>
<p class="zh-tr" align="center">Overlap IO 与计算有效吸收了额外的 memory traffic，因此 IO cost 的增加并不按比例转化为 runtime 增加。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>"亚比例"为什么是关键判据</strong>
<p>如果 epilogue 完全串行（MMA 完才 epilogue），HBM +24% 应该让 runtime +24%、TFLOPS −24%。实测 −11% ⇒ 多出来的 13% IO 完全藏在 MMA 后面。如果 IO 与 MMA 100% overlap，"瓶颈"就只看 $\max(\text{MMA time}, \text{IO time})$；这里两者打平（Tensor Core 88%、TMEM 88%）⇒ 接近最优 overlap 状态。</p>
<p>要做到这一点的硬件前提：(1) TMEM 双 buffer 给"MMA 写一个 stage、epilogue drain 另一个"提供物理基础；(2) <code>st.async.release.global</code> 让 epilogue 的三次 store（dH/A'/dS）不阻塞下一 stage 的 MMA。</p>
<p>SonicMoE 的特殊安排：epilogue warp 之间也分工 —— 一个 warp 专门 TMA-load <code>h</code>（它是 epilogue 内部的 producer），其他 warp 做 SwiGLU 重算 / dSwiGLU jacobian / colvec_reduce。这是嵌套了"epilogue 内部的 producer-consumer"。普通 CUTLASS epilogue 做不到，QuACK 的 warp specialization 抽象支持这个。</p>
</div> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/32k-4k-1.5k-128-8-dH-memory-chart.png" width="47%"> &nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/32k-4k-1.5k-128-8-gemm-alone-memory-chart.png" width="48%"></p>
<p class="zh-tr" align="center">5. Benchmark 结果</p> <p align="center"><em>Figure: Nsight Compute Profiling of SonicMoE's dH kernel Grouped GEMM with 4 epilogue ops (left column) vs. Grouped GEMM alone (right column) of Qwen3-235B-A22B-Thinking-2507 (microbatch size=32k) on B300 GPUs. The top row is the achieved throughput of Tensor Pipe (MMA) and DRAM at kernel runtime, and the bottom row shows the transferred bytes on hardware units. </em></p>
<p class="zh-tr" align="center">在 B300 GPU 上对 SonicMoE 与多个 baseline 做评估。我们 benchmark 单层 MoE 的前后向 pass，配置改编自开源 7B 到 685B MoE，然后专门在 7B MoE 上做 kernel 级时间分解。</p> <blockquote> <p><strong>Overlapping IO with computation effectively absorbs the additional memory traffic, so the increase in IO cost does not translate proportionally into increased runtime.</strong></p>
<p class="zh-tr">6 个开源 MoE 配置上的前后向 TFLOPS</p> </blockquote> <h2 id="5-benchmark-results">5. Benchmark Results</h2>
<h2 class="zh-h" id="5-benchmark-results">下图给出 6 个真实开源 MoE 配置（7B 到 685B MoE 模型）上的前后向 TFLOPS。</h2> <p>We evaluate SonicMoE against multiple baselines on B300 GPUs. We benchmark the forward and backward pass of a single MoE layer with configurations adapted from open-source 7B to 685B MoE, and we then profile kernel-level time breakdown on 7B MoE specifically.</p>
<p class="zh-tr">图：B300 上 6 个真实 MoE 配置的 forward（左）与 backward（右）TFLOPS。从左到右：OLMoE-1B-7B-0125、gpt-oss-20b、Kimi-Linear-48B-A3B-Base、Qwen3-Next-80B-A3B-Thinking、Qwen3-235B-A22B-Thinking-2507、DeepSeek-V3.2-Exp。Triton 官方 example 不支持反向，也不支持 Qwen3-Next-80B forward 的 K=10。</p> <h3 id="forward-and-backward-tflops-of-6-open-source-moe-configs">Forward and Backward TFLOPS of 6 Open-source MoE Configs</h3>
<h3 class="zh-h" id="forward-and-backward-tflops-of-6-open-source-moe-configs">Baseline：</h3> <p>The figure below shows forward and backward TFLOPS across six real open-source MoE configurations, ranging from a 7B to a 685B MoE model.</p>
<p class="zh-tr">结果：</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/real_moe_benchmark-B300.png" width="100%"></p> <p align="center"><em>Figure: Forward (left) and backward (right) TFLOPS on B300 for 6 real MoE configurations. From left to right: OLMoE-1B-7B-0125, gpt-oss-20b, Kimi-Linear-48B-A3B-Base, Qwen3-Next-80B-A3B-Thinking, Qwen3-235B-A22B-Thinking-2507, and DeepSeek-V3.2-Exp. Triton official example does not support backward pass, nor K=10 for Qwen3-Next-80B forward pass. </em></p>
<p class="zh-tr" align="center">SonicMoE 在所有配置上一致领先。6 个配置平均：forward / backward TFLOPS 比 DeepGEMM baseline 高 54% / 35%；forward 比 Triton 官方 example 高 21%。在所有配置上 SonicMoE 都对 ScatterMoE 与 MoMoE 有决定性优势（往往达到 2× TFLOPS）。</p> <h4 id="baselines">Baselines:</h4>
<h4 class="zh-h" id="baselines">Profiling 时间分解</h4> <table data-toggle="table" class="table-hover"> <thead> <tr> <th>Baseline</th> <th>Description</th> </tr> </thead> <tbody> <tr> <td><strong>ScatterMoE</strong></td> <td> <a href="https://github.com/open-lm-engine/accelerated-model-architectures/blob/main/xma/layers/moe/triton_implementation/__init__.py" rel="external nofollow noopener" target="_blank">OpenLM Engine version</a> (same kernel code, slightly different API).</td> </tr> <tr> <td><strong>MoMoE</strong></td> <td> <a href="https://github.com/tilde-research/MoMoE-impl" rel="external nofollow noopener" target="_blank">Official implementation</a> with shared experts disabled and expert bias adjustment removed.</td> </tr> <tr> <td><strong>DeepGEMM</strong></td> <td>DeepGEMM’s <a href="https://github.com/deepseek-ai/DeepGEMM/blob/d30fc36c8f229f4f873b90a492f6e19e6e610923/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp#L124" rel="external nofollow noopener" target="_blank">SM100 varlen-M</a> and <a href="https://github.com/deepseek-ai/DeepGEMM/blob/d30fc36c8f229f4f873b90a492f6e19e6e610923/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp#L233" rel="external nofollow noopener" target="_blank">varlen-K</a> BF16 Grouped GEMM, paired with a separate optimized gather kernel and <code class="language-plaintext highlighter-rouge">torch.compile</code> for all activation and expert aggregation kernels. This represents the throughput a practitioner would achieve by integrating DeepGEMM as a drop-in Grouped GEMM library.</td> </tr> <tr> <td><strong>Triton official example</strong></td> <td>Adapted from <a href="https://github.com/triton-lang/triton/blob/7d0756121cc95d6971112fc5c1fa99107b892444/python/triton_kernels/bench/bench_mlp.py#L53" rel="external nofollow noopener" target="_blank">bench_mlp.py</a> with expert parallelism disabled.</td> </tr> </tbody> </table> <h4 id="results">Results:</h4>
<h4 class="zh-h" id="results">下面的 runtime 分解把加速来源具体化。SonicMoE 中 forward 的 "gather X" 段与 backward 的 "gather dO 与 X" 段被吸收进 GEMM bar —— 这是相对 DeepGEMM-built baseline 的一大主要加速来源（后者也有优化的 Grouped GEMM 但需要单独 gather kernel）。</h4> <p><strong>SonicMoE consistently leads on all configurations</strong>. On average across 6 configs, SonicMoE achieves 54%/35% higher forward/backward TFLOPS than DeepGEMM baseline, and 21% higher forward TFLOPS than triton official example. <strong>SonicMoE has a decisive advantage (often achieving <em>double</em> TFLOPS) over the ScatterMoE and MoMoE baselines across all configs.</strong></p>
<p class="zh-tr">尽管 Triton 官方 example 也有 gather fusion 且不存 $H$（推理向，无需 cache activation），SonicMoE 在 forward 三个 kernel 上仍然全部更快。原因是 SonicMoE 用了带 CLC tile scheduler 与 2CTA MMA 的更快 Grouped GEMM，且 expert aggregation kernel 经过重度优化。详见附录。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>把 +54% 拆成贡献分量</strong>
<table style="border-collapse:collapse;font-size:13px;margin:8px 0;">
<thead><tr><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">来源</th><th style="padding:4px 10px;border:1px solid #ccc;background:#eee;">对 forward 提升的贡献</th></tr></thead>
<tbody>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">消除独立 gather kernel（gather fusion，省一个 kernel + 4 GB HBM）</td><td style="padding:4px 10px;border:1px solid #ccc;">~25-30%</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">2CTA UMMA + B-tile multicast（AI 翻倍）</td><td style="padding:4px 10px;border:1px solid #ccc;">~7-10%</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">CLC dynamic schedule（消灭 MoE 长尾）</td><td style="padding:4px 10px;border:1px solid #ccc;">~3-5%</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">L2 cache locality（fewer HBM misses）</td><td style="padding:4px 10px;border:1px solid #ccc;">~5-8%</td></tr>
<tr><td style="padding:4px 10px;border:1px solid #ccc;">SMEM swizzle / warp layout 微调</td><td style="padding:4px 10px;border:1px solid #ccc;">~3-5%</td></tr>
</tbody></table>
<p>Backward +35% 中，dS contraction 重排（消灭 dY 的 GEMM）单项约 15-20%，其余来自上面的硬件项。</p>
<p><b>2× ScatterMoE / MoMoE</b> 的来源更宽：那两家是 Hopper 时代的 monolithic kernel，没用上 UMMA / TMEM / 2CTA / CLC / async store，scatter 用 atomic_add 阻塞 epilogue。所以"Blackwell 硬件 + SonicMoE 算法"在 2× 量级很合理。</p>
</div> <h3 id="profiling-time-breakdown">Profiling Time Breakdown</h3>
<h3 class="zh-h" id="profiling-time-breakdown">图：B300 上 7B OLMoE-sized MoE（$T=32768, d=2048, n=1024, E=64, K=8$）的 SonicMoE 与 baseline runtime 分解。其他 baseline 描述见 arXiv 论文 Figure 5 caption。本配置下 SonicMoE 的主要加速来自 gather fusion，更快的 GEMM 再贡献 ~10%。图中将 TFLOPS 缩写为 "TF/s"。</h3> <p>The runtime breakdown below makes the speedup concrete. The “gather <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="142" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container>” segment in the forward pass and “gather <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="143" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container> and <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="144" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container>” segment in the backward pass are absorbed into the GEMM bars for SonicMoE, and this constitutes one major source of speedup over the DeepGEMM-built baseline, which also has optimized Grouped GEMM but requires a separate gather kernel.</p>
<p class="zh-tr">结论</p> <p>We note that <strong>although Triton official example has gather fusion and <em>does not</em> store <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="145" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>H</mi></math></mjx-assistive-mml></mjx-container> (as it is inference-oriented with no need of caching activation), SonicMoE is still faster for all three kernels during forward pass</strong>. This is because SonicMoE employs a faster Grouped GEMM implementation with the CLC tile scheduler and 2CTA MMA, and the expert aggregation kernel is heavily optimized. Please refer to the appendix for more details.</p>
<p class="zh-tr">SonicMoE 起源于一个简单观察：业界正在构建越来越 fine-grained、越来越 sparse 的 MoE，而现有 kernel 并非为该 regime 设计。从 Mixtral 到 Kimi K2.5 大约 2 年，granularity 提升 9×，activation ratio 下降 12×；每一步都让算术强度更糟、activation memory 更大。我们需要重新审视基础设施设计 blueprint 来拥抱这一 MoE 趋势 —— SonicMoE 是我们的回应之一。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/moe_breakdown_fwd_bwd-B300.png" width="100%"></p> <p align="center"><em>Figure: Runtime breakdown of SonicMoE vs baselines on B300 for a 7B OLMoE-sized MoE (T=32768, d=2048, n=1024, E=64, K=8). Detailed descriptions of other baselines can be found in the caption of Figure 5 of our arXiv paper. On this config, SonicMoE's major speedup comes from the gather fusion, and the faster GEMM delivers another 10% speedup. We abbreviate TFLOPS as "TF/s" in the figure. </em></p>
<p class="zh-tr" align="center">Activation memory 高效且 IO-aware 的算法设计。通过重设计反向 pass 来避免 cache 任何 $O(TKd)$ 张量，SonicMoE 的单层 activation memory 与 expert granularity 解耦 —— 与同等 active 参数 dense 模型相同，且无任何 GEMM recomputation。同样的算法重排消灭多次大型 HBM round-trip，剩余 IO cost 通过 Hopper 与 Blackwell 的硬件异步性大量藏在 MMA 计算后面。</p> <h2 id="conclusion">Conclusion</h2>
<h2 class="zh-h" id="conclusion">可扩展的软件抽象 + hardware-aware 优化。SonicMoE 所有 kernel 都是建在 QuACK 之上的同一种共享结构的实例。该抽象把架构特定行为局限到少量 override，epilogue fusion 逻辑与 GEMM interface 不变 —— 让原型新模型架构 / benchmark 新硬件特性都能快速迭代。</h2> <p>SonicMoE started from a simple observation: the field is building MoEs that are more fine-grained and sparser with every generation, and existing kernels were not designed for that regime. Roughly 2 years from Mixtral to Kimi K2.5 represent a 9× increase in granularity and a 12× drop in activation ratio, and every step of that journey makes the arithmetic intensity worse and the activation memory larger. <strong>We need to re-visit our infrastructure design blueprint to embrace this MoE model trend, and SonicMoE is one of our answers.</strong></p>
<p class="zh-tr">未来方向。最直接的扩展是 expert parallelism：IO-aware 设计原则可直接迁移到 intra-/inter-node 场景，那里网络带宽比 HBM 更受限。之后我们计划添加 MXFP8 与 MXFP4 支持。最后，下一代 GPU（Rubin）会带来新硬件原语 —— 有了这套抽象，预计移植工作量不会超过 Hopper-to-Blackwell 那一次。</p> <ul> <li> <p><strong>Activation memory-efficient and IO-aware algorithm design.</strong> By redesigning the backward pass to avoid caching any <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="146" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>O</mi><mo stretchy="false">(</mo><mi>T</mi><mi>K</mi><mi>d</mi><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container>-sized tensor, SonicMoE’s per-layer activation memory is independent of expert granularity — matching a dense model with the same activated parameter count, without any GEMM recomputation. The same algorithmic reordering eliminates multiple large HBM round-trips, and the remaining IO costs are largely hidden behind MMA computation through hardware asynchrony on both Hopper and Blackwell GPUs.</p>
<p class="zh-tr">如何引用本博客</p> </li> <li> <p><strong>Extensible software abstraction with hardware-aware optimization.</strong> All of SonicMoE’s kernels are instances of one shared structure built on QuACK. This abstraction confines architecture-specific behavior to localized overrides while leaving the epilogue fusion logic and the GEMM interface untouched. This enables fast iteration for prototyping new model architectures and benchmarking new hardware features.</p>
<p class="zh-tr">如果 SonicMoE 在你的研究或开发中有帮助，欢迎引用：</p> </li> </ul> <p><strong>Future directions.</strong> The most immediate extension is expert parallelism: the IO-aware design principles transfer directly to the intra-node and inter-node setting, where network bandwidth is even more constraining than HBM. After that, we plan to add MXFP8 and MXFP4 support. Finally, the next GPU generation (Rubin) will bring new hardware primitives, and with the abstraction in place, we expect the port to require no more work than the Hopper-to-Blackwell migration did.</p> <h2 id="citing-this-blogpost">Citing this blogpost</h2> <p>If you find SonicMoE helpful in your research or development, please consider citing us:</p> <div class="language-plaintext highlighter-rouge"><div class="highlight"><div class="code-display-wrapper"><pre class="highlight"><code>@article{guo2025sonicmoe,
  title={SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations},
  author={Guo, Wentao and Mishra, Mayank and Cheng, Xinle and Stoica, Ion and Dao, Tri},
  journal={arXiv preprint arXiv:2512.14080},
  year={2025}
}
</code></pre><button class="copy" type="button" aria-label="Copy code to clipboard"><i class="fa-solid fa-clipboard"></i></button></div></div></div> <h2 id="references">References</h2> <p>[1] Yang, Haoqi, et al. “Faster moe llm inference for extremely large models.” arXiv preprint arXiv:2505.03531 (2025).</p> <p>[2] Michael Diggin. “Implementing a Split-K Matrix Multiplication Kernel in Triton.” https://medium.com/@michael.diggin/implementing-a-split-k-matrix-multiplication-kernel-in-triton-7ad93fe4a54c</p> <p>[3] NVIDIA CUTLASS Documentation. “Blackwell Cluster Launch Control.” https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/blackwell_cluster_launch_control.html</p> <p>[4] Modular. “Matrix Multiplication on NVIDIA’s Blackwell Part 3: The Optimizations Behind 85% of SOTA Performance.” https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance</p> <p>[5] PyTorch Blog. “Enabling Cluster Launch Control with TLX.” https://pytorch.org/blog/enabling-cluster-launch-control-with-tlx/</p>
<p class="zh-tr">附录</p> <p>[6] Alex Armbuster. “How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores.” https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html</p>
<p class="zh-tr">下面收集支持正文的实现对比与消融研究：cp.async vs. TMA gather4 for gather fusion、gather fusion 的 L2 cache locality 分析、GEMM + gather-and-sum vs. GEMM with scatter fusion + sum 的 expert aggregation 设计选择。</p> <p>[7] K.V. Nagesh. “NVIDIA Blackwell Architecture: A Deep Dive into the Next Generation of AI Computing.” https://medium.com/@kvnagesh/nvidia-blackwell-architecture-a-deep-dive-into-the-next-generation-of-ai-computing-79c2b1ce3c1b</p>
<p class="zh-tr">消融研究</p> <p><br></p> <p><br></p> <p><br></p> <p><br></p> <p><br></p> <h2 id="appendix">Appendix</h2>
<h2 class="zh-h" id="appendix">cp.async vs. TMA gather4 for gather fusion</h2> <p>Below, we collect implementation comparisons and ablation studies that support the main text. We present a few ablation studies: cp.async vs. TMA gather4 for gather fusion, L2 cache locality analysis of gather fusion, and the design choice between GEMM + gather-and-sum vs. GEMM with scatter fusion + sum for expert aggregation.</p>
<p class="zh-tr">我们先在 <code>cp.async</code> 路径上 autotune 出最佳 GEMM 配置（tile shape、tile scheduler 类型等），然后原地切到 TMA gather。下图发现两种机制 TFLOPS 接近（大多数 case 差异 < 2%）。即便如此，我们仍把 "用 TMA gather 还是 cp.async gather" 作为 kernel runtime 的 autotunable 配置。</p> <h3 id="ablation-studies">Ablation Studies</h3>
<h3 class="zh-h" id="ablation-studies">图：B300 上 forward up-proj（$M$ 维 gather）与 backward dW1 kernel（$K$ 维 gather）的 <code>cp.async</code> vs. TMA gather TFLOPS。百分比为 TMA gather 相对 <code>cp.async</code> 的 TFLOPS 差异。</h3> <h4 id="cpasync-vs-tma-gather4-for-gather-fusion"> <code class="language-plaintext highlighter-rouge">cp.async</code> vs. TMA gather4 for gather fusion</h4>
<h4 class="zh-h" id="cpasync-vs-tma-gather4-for-gather-fusion">Gather Fusion 的 L2 Cache Locality</h4> <p>We first autotune on the best GEMM configs (tile shape, tile scheduler type, etc.) with <code class="language-plaintext highlighter-rouge">cp.async</code>, and then we switch in-place to TMA gather. In the following figure, we find that these two mechanisms deliver similar TFLOPS (diff &lt; 2% for most cases). Nevertheless, we add whether to use TMA gather or cp.async gather as an autotunable configuration at kernel runtime.</p>
<p class="zh-tr">对比 gather fusion 与「独立 gather kernel 把输入预排成 contiguous buffer 再喂 Grouped GEMM」。下面 NCU memory chart 显示一个 varlen-M Grouped GEMM kernel 用 gather fusion（左）vs. 用 pre-gathered contiguous 输入（右）。L2→SMEM traffic 几乎一样（17.74 GB），但 gather fusion 的 HBM load traffic 更少（2.20 vs 2.68 GB），L2 hit rate 更高（74.9% vs 66.3%）。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/cpasync-tma-gather-comparison.png" width="100%"></p> <p align="center"><em>Figure: `cp.async` vs. TMA gather TFLOPS for forward pass up-proj (gather on M dim) and backward dW1 kernel (gather on K dim) kernels on B300 GPUs. Percentages indicate the relative TFLOPS difference of TMA gather over `cp.async`. </em></p>
<p class="zh-tr" align="center">图：MoE 大小 $(T,d,n,E,K)=(32768,2048,512,256,32)$ 在 up-proj forward pass 时 varlen-M Grouped GEMM 的 NCU memory chart。左：gather fusion 用 <code>cp.async</code>。右：contiguous TMA load 用 pre-gathered 输入。两者用相同的 tile shape、scheduler 配置、L2 swizzling 模式。</p> <h4 id="l2-cache-locality-with-gather-fusion">L2 Cache Locality with Gather Fusion</h4>
<h4 class="zh-h" id="l2-cache-locality-with-gather-fusion">原因：gather fusion 的 source tensor（$X$ 或 $dO$）大小通常是 $T\times d$，比 pre-gathered 的 $T\times K\times d$ 小 $K$ 倍。expert granularity 增大时 $K$ 等比例增长，pre-gathered tensor 可能超过 GPU 的 L2 cache 容量（B300 上 192 MB）。一旦超过，数据请求 miss L2 走 HBM。gather fusion 避免这一点：从 compact 的原始 tensor 读，更可能 stay resident 在 L2 中。</h4> <p>We compare the gather fusion against running a separate gather kernel to pre-arrange the inputs into a contiguous buffer before feeding into the Grouped GEMM kernel. The Nsight Compute memory charts below show a varlen-M Grouped GEMM kernel with gather fusion (left) and with pre-gathered contiguous inputs (right). Despite nearly identical L2-&gt;SMEM traffic (17.74 GB), the gather fusion (left figure) shows less HBM load traffic (2.20 vs 2.68 GB) and higher L2 cache hit rate (74.9% vs 66.3%).</p>
<p class="zh-tr">这一优势随 expert granularity 复合放大。Gathered $X$ 与 gathered $dO$ 是 SonicMoE 6 个 Grouped GEMM kernel 中 4 个的输入，都是 $O(TKd)$ 大小且随 $K$ 线性增长。下图证实跨三个模型 family 的趋势：随 granularity 增大（每列从左到右），contiguous 路径的 HBM load traffic 增长更快，相对 gather fusion 的 L2 hit rate 也下降更多。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/olmoe-512-gather-memory-chart.png" width="48%">&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/olmoe-512-TMA-memory-chart.png" width="48.5%"></p>
<p class="zh-tr" align="center">图：B300 上 gather fusion vs. contiguous load 在不同 expert granularity 下的 HBM load 字节（顶行）与 device L2 cache hit rate（底行）。顶行标注显示 contiguous 路径相对 gather fusion 的 HBM load 绝对 / 相对增量；底行标注显示 gather fusion 的 L2 hit rate 优势。</p> <p align="center"><em>Figure: Nsight Compute memory chart for varlen-M Grouped GEMM during up-proj forward pass for MoE with size (T, d, n, E, K) = (32768, 2048, 512, 256, 32). Left: gather fusion with `cp.async`. Right: contiguous TMA load with pre-gathered inputs. Both use the same tile shape, scheduler configuration, and L2 swizzling pattern. </em></p>
<p class="zh-tr" align="center">Expert Aggregation Bandwidth</p> <p>This is because gather fusion’s source tensor (<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="147" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> or <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="148" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container>) often has size <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="149" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="3"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>T</mi><mo>×</mo><mi>d</mi></math></mjx-assistive-mml></mjx-container>, which is <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="150" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi><mo>×</mo></math></mjx-assistive-mml></mjx-container> smaller than the pre-gathered tensor of size <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="151" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="3"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n" space="3"><mjx-c class="mjx-cD7"></mjx-c></mjx-mo><mjx-mi class="mjx-i" space="3"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>T</mi><mo>×</mo><mi>K</mi><mo>×</mo><mi>d</mi></math></mjx-assistive-mml></mjx-container>. As expert granularity increases, <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="152" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi></math></mjx-assistive-mml></mjx-container> grows proportionally, and the pre-gathered tensor can exceed the GPU’s L2 cache capacity (192 MB on B300). When this happens, the data request will miss L2 and be served from HBM. Gather fusion avoids this: it reads from the compact original tensor, which is more likely to stay resident in L2 cache.</p>
<p class="zh-tr">SonicMoE 的 expert aggregation kernel：每个 token 并行 gather Grouped GEMM 结果并求和。无 GEMM、纯 memory-bound。第一版基于 CuteDSL 实现，后切到纯 Triton 实现（autotune 方便）。Hopper 上接近 peak memory bandwidth，下面验证 Blackwell 上的性能：</p> <p>This advantage compounds with expert granularity. Gathered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="153" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D44B TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math></mjx-assistive-mml></mjx-container> and gathered <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="154" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi><mi>O</mi></math></mjx-assistive-mml></mjx-container>, which are inputs to four of SonicMoE’s six Grouped GEMM kernels, are both <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="155" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D442 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c28"></mjx-c></mjx-mo><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D447 TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D451 TEX-I"></mjx-c></mjx-mi><mjx-mo class="mjx-n"><mjx-c class="mjx-c29"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>O</mi><mo stretchy="false">(</mo><mi>T</mi><mi>K</mi><mi>d</mi><mo stretchy="false">)</mo></math></mjx-assistive-mml></mjx-container>-sized and grow linearly with <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" tabindex="0" ctxtmenu_counter="156" style="font-size: 119.4%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D43E TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi></math></mjx-assistive-mml></mjx-container>. The figures below confirm the trend across three model families: as granularity increases (from left to right on each column), the contiguous path’s HBM load traffic grows faster and its L2 hit rate drops further relative to gather fusion.</p>
<p class="zh-tr">图：B300 上 1.4B、7B、30B、120B MoE 配置的 expert aggregation kernel memory bandwidth。SonicMoE 的 gather-and-sum kernel（蓝）在每个 scale 都接近 triton 上界（灰，<code>tl.load</code> 与 TMA 的最大值）。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/gather-l2-analysis.png" width="100%"></p> <p align="center"><em>Figure: HBM load bytes (top row) and device L2 cache hit rate (bottom row) for gather fusion vs. contiguous load across varying expert granularity on B300 GPUs. Annotations on the top row show the absolute and relative HBM load increase of the contiguous path over gather fusion. Annotations on the bottom row show the L2 hit rate advantage of gather fusion. </em></p>
<p class="zh-tr" align="center">出乎意料地，这个 Triton 实现在 Blackwell GPU（B300）上仍然性能足够好。该 kernel 在大多数配置上超过 6.5 TB/s（85%+ peak），达到 contiguous 输入上优化求和 kernel 的 98%。我们也发现这个简单 aggregation kernel 比从 Gluon 官方 example 改编的 Gluon TMA gather-and-sum 平均高 5%。这进一步说明 cp.async gather 不比 TMA gather 差。</p> <h4 id="expert-aggregation-bandwidth">Expert Aggregation Bandwidth</h4>
<h4 class="zh-h" id="expert-aggregation-bandwidth">GEMM + gather-and-sum vs. GEMM with scatter + sum aggregation</h4> <p>In SonicMoE’s expert aggregation kernel, each token will gather the Grouped GEMM results and sum over them in parallel. No GEMM is involved and this is a memory-bound kernel. The first version was implemented on CuteDSL, but we later switched to a pure Triton implementation due to the convenience of autotuning. This kernel achieves close-to-peak memory bandwidth on Hopper; here we validate its performance on Blackwell:</p>
<p class="zh-tr">图：每个 token 存储与聚合结果的可能策略。SonicMoE 选第一种（左）—— 每个 expert 在 GEMM epilogue 直接通过 TMA 存 contiguously-packed 输出；expert aggregation kernel 中每个 token gather 并求和被激活的 expert 输出。ScatterMoE 与 MoMoE 选中间方案 —— epilogue 中 fuse HBM store 与 scatter，之后跑求和 kernel。也可以在 epilogue 中 fuse atomic add 来绕开 expert aggregation kernel（右图）。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/reduction_benchmark-B300.png" width="100%"></p> <p align="center"><em>Figure: Expert aggregation kernel memory bandwidth on B300 across 1.4B, 7B, 30B, and 120B MoE configurations. SonicMoE's gather-and-sum kernel (blue) approaches the triton upper bound (grey, max of tl.load and TMA) at every scale. </em></p>
<p class="zh-tr" align="center">在 Hopper GPU 上，SonicMoE 做了一个非常规设计选择：不把 scatter 与 GEMM fuse，而是与 aggregation 一起做。我们之前在 Hopper 上的消融发现：scatter fusion 在 Hopper 上需要的 synchronous <code>st.global</code> PTX 指令对 fine-grained MoE 配置会让 TFLOPS 降 20%。</p> <p><strong>Surprisingly, we find this Triton implementation still performs well enough on Blackwell GPUs (B300). This kernel surpasses 6.5 TB/s (85%+ peak) across most configs, achieving 98% of an optimized summation kernel on contiguous inputs.</strong> We also find this simple aggregation kernel outperforms the <a href="https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/09-tma-gather-scatter.py" rel="external nofollow noopener" target="_blank">Gluon TMA gather-and-sum, adapted from Gluon official example</a> implementation by 5% on average. This further suggests that gather with <code class="language-plaintext highlighter-rouge">cp.async</code> is not worse than TMA gather.</p>
<p class="zh-tr">IO-aware 设计只在算法意图与硬件执行语义被一起推理时才会浮现。</p>
<div class="deep-dive"><div class="dd-label">深度解读</div>
<strong>为什么 atomic_add scatter 是诱人陷阱</strong>
<p>"在 epilogue 里 fuse atomic_add 直接写到 output，省掉 expert aggregation kernel"看起来最优雅 —— 一发 kernel 解决，没有中间张量。但实际：</p>
<ol>
<li><b>atomic_add 的串行化</b>：当多个 CTA 同时往同一 token 的同一段累加，HW lock 让它们串行。MoE 里同一 token 的 K 个 expert 输出几乎同时被算出（不同 CTA），冲突频繁。</li>
<li><b>非确定性</b>：浮点加法不结合，atomic_add 顺序不固定 ⇒ 训练 reproducibility 受影响（同一 weight 不同时刻 forward 出来略不同 loss）。</li>
<li><b>影响 epilogue pipeline</b>：atomic_add 是同步 store，没法和 async store 那样和下一个 tile 的 MMA overlap。</li>
</ol>
<p>SonicMoE 选的策略把 scatter 从"原子写"改成"先各自 contiguous 写、再单独 gather+sum kernel"：(a) 完全确定性、(b) GEMM epilogue 用 async store 不阻塞 pipeline、(c) gather+sum kernel 独立 autotune 到 6.5 TB/s。</p>
</div> <h4 id="gemm--gather-and-sum-vs-gemm-with-scatter--sum-aggregation">GEMM + gather-and-sum vs. GEMM with scatter + sum aggregation</h4>
<h4 class="zh-h" id="gemm--gather-and-sum-vs-gemm-with-scatter--sum-aggregation">Blackwell 上的新异步 Scatter Store 指令</h4> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/expert-agg.png" width="100%"></p> <p align="center"><em>Figure: Possible strategies for storing the results and aggregating the results for each token. SonicMoE chooses the first strategy (left) in which each expert directly stores contiguously-packed outputs via TMA in the GEMM epilogue. In the expert aggregation kernel, each token gathers and sums over activated expert outputs. ScatterMoE and MoMoE (middle) choose to fuse HBM store with scatter in epilogue and launch a summation kernel afterwards. It is also possible to fuse atomic add in the epilogue to circumvent the requirement of an expert aggregation kernel as the right subfigure illustrated. </em></p>
<p class="zh-tr" align="center">然而 Blackwell 引入了多条 asynchronous store 指令：(1) <code>st.async.release.global</code>；(2) TMA scatter4。GEMM + gather-and-sum 相对 GEMM w. scatter fusion + sum 的优势变得不那么明显 —— 不再有 Hopper 上那种 scatter fusion 的同步 IO 问题。即便如此：(1) gather-and-sum 与 contiguous 求和 kernel 的带宽差距只有 0.98×；(2) 我们预期 GEMM with TMA 不会比 GEMM with TMA scatter4 或 <code>st.async</code> 慢；所以在 Blackwell 上保留 SonicMoE 原设计。</p> <p>On Hopper GPUs, SonicMoE makes an unconventional design choice that we <em>do not</em> fuse scatter with GEMM. Instead, we perform this task alongside the aggregation. <strong>We previously ablated on Hopper GPUs and identified that the synchronous <code class="language-plaintext highlighter-rouge">st.global</code> PTX instruction required for scatter fusion on Hopper would degrade TFLOPS by 20% for fine-grained MoE configs.</strong></p>
<p class="zh-tr">对比 varlen-M Grouped GEMM w. TMA + gather-and-sum vs. varlen-M Grouped GEMM w. TMA scatter + sum，两者都改编自 Triton 官方 Grouped GEMM example。grouped gemm w. TMA + gth-and-sum 在 down-proj forward epilogue 中把 Grouped GEMM 结果存到跨 expert 的 contiguously-packed tensor，每个 token 在 single fused 操作中 gather 并求和对应的 expert 输出。grouped gemm w. TMA sct + sum 则在 epilogue 中通过 TMA scatter 结果，之后另起 contiguous 求和 kernel。</p> <blockquote> <p>An IO-aware design emerges only when algorithmic intent and hardware execution semantics are reasoned about together.</p>
<p class="zh-tr">声明：本消融研究中的 Grouped GEMM kernel 用 Triton 实现，低层优化（如未用 2CTA MMA）少于 SonicMoE 的 Grouped GEMM；但仍能给出 GEMM with TMA 与 GEMM with TMA scatter4 的相对性能对比的洞察。</p> </blockquote> <h5 id="new-asynchronous-scatter-store-instructions-on-blackwell">New Asynchronous Scatter Store Instructions on Blackwell</h5>
<h5 class="zh-h" id="new-asynchronous-scatter-store-instructions-on-blackwell">图：B300 上 forward pass 中 varlen-M Grouped GEMM 与 expert aggregation kernel 的吞吐。第一行：透明柱报告 Grouped GEMM TFLOPS，不透明柱报告 gemm-and-aggregation TFLOPS；第二行：对比 gather-and-sum 与 contiguous 求和 kernel 的 expert aggregation 带宽。</h5> <p>However, Blackwell introduces multiple asynchronous store instructions: (1) <code class="language-plaintext highlighter-rouge">st.async.release.global</code> and (2) TMA scatter4. <strong>The advantage of GEMM + gather-and-sum over GEMM w. scatter fusion + sum becomes less apparent as we no longer run into the synchronous IO issue for GEMM w. scatter fusion on Hopper.</strong> Even so, as we (1) do not observe major bandwidth degradation (0.98x) of gather-and-sum compared with contiguous summation kernel and (2) expect GEMM with TMA to be no slower than GEMM with TMA scatter4 or <code class="language-plaintext highlighter-rouge">st.async</code>, we do not change SonicMoE’s design choice on Blackwell.</p>
<p class="zh-tr">在第一行：</p> <p>We perform an ablation comparing varlen-M Grouped GEMM w. TMA + gather-and-sum against varlen-M Grouped GEMM w. TMA scatter + sum, adapting the official Triton Grouped GEMM example for both. The <code class="language-plaintext highlighter-rouge">grouped gemm w. TMA + gth-and-sum</code> approach stores Grouped GEMM results into a contiguously-packed tensor across all experts during the down-projection forward epilogue, where each token gathers and sums its corresponding expert outputs in a single fused operation. The <code class="language-plaintext highlighter-rouge">grouped gemm w. TMA sct + sum</code> approach instead scatters results via TMA during the epilogue and applies a separate contiguous summation kernel afterwards.</p>
<p class="zh-tr">GEMM-only TFLOPS（透明柱）：grouped gemm w. TMA 仍比 grouped gemm w. TMA sct 高 5%。</p> <p><em>Disclaimer: the Grouped GEMM kernel in this ablation study is implemented with triton with fewer low-level optimizations (e.g. without 2CTA MMA) than SonicMoE’s Grouped GEMM, but it still provides insight on the relative performance comparison between GEMM w. TMA and GEMM w. TMA scatter4.</em></p>
<p class="zh-tr">GEMM-and-aggregation TFLOPS（不透明柱）：grouped gemm w. TMA + gth-and-sum 仍比 grouped gemm w. TMA sct + sum 高 3%。</p> <p align="center"><img src="https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/media/triton_example_grouped_gemm_expert_agg.png" width="100%"></p> <p align="center"><em>Figure: Throughput of varlen-M Grouped GEMM and expert aggregation kernel on B300 GPUs during forward pass. In the first row, we report the Grouped GEMM TFLOPS on transparent bars and the gemm-and-aggregation TFLOPS on opaque bars. In the second row, we compare the expert aggregation bandwidth between gather-and-sum and a contiguous sum kernel.</em></p>
<p class="zh-tr" align="center">在第二行，我们已经知道 gth-and-sum 比 sum 仅低 2% 带宽。</p> <p>In the first row,</p>
<p class="zh-tr">尽管这个 3% gap 远小于 Hopper 上 20% 的 gap，仍然验证了 SonicMoE 在 Blackwell 上的设计选择。</p> <ul> <li> <p><strong>GEMM-only TFLOPS</strong> (transparent bars): <code class="language-plaintext highlighter-rouge">grouped gemm w. TMA</code> still has 5% higher TFLOPS than <code class="language-plaintext highlighter-rouge">grouped gemm w. TMA sct</code></p> </li> <li> <p><strong>GEMM-and-aggregation TFLOPS</strong> (opaque bars): <code class="language-plaintext highlighter-rouge">grouped gemm w. TMA + gth-and-sum</code> still has 3% higher TFLOPS than <code class="language-plaintext highlighter-rouge">grouped gemm w. TMA sct + sum</code></p> </li> </ul> <p>In the second row, we already know that <code class="language-plaintext highlighter-rouge">gth-and-sum</code> only has 2% less bandwidth than <code class="language-plaintext highlighter-rouge">sum</code>.</p> <p>Although this 3% gap is much smaller than the prior gap on Hopper GPUs (20%), it still validates SonicMoE’s design on Blackwell GPUs.</p> </d-article> <d-appendix>
<style>

d-appendix {
  contain: layout style;
  font-size: 0.8em;
  line-height: 1.7em;
  margin-top: 60px;
  margin-bottom: 0;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  color: rgba(0,0,0,0.5);
  padding-top: 60px;
  padding-bottom: 48px;
}

d-appendix h3 {
  grid-column: page-start / text-start;
  font-size: 15px;
  font-weight: 500;
  margin-top: 1em;
  margin-bottom: 0;
  color: rgba(0,0,0,0.65);
}

d-appendix h3 + * {
  margin-top: 1em;
}

d-appendix ol {
  padding: 0 0 0 15px;
}

@media (min-width: 768px) {
  d-appendix ol {
    padding: 0 0 0 30px;
    margin-left: -30px;
  }
}

d-appendix li {
  margin-bottom: 1em;
}

d-appendix a {
  color: rgba(0, 0, 0, 0.6);
}

d-appendix > * {
  grid-column: text;
}

d-appendix > d-footnote-list,
d-appendix > d-citation-list,
d-appendix > distill-appendix {
  grid-column: screen;
}

</style>

 <d-footnote-list style="display: none;">
<style>

d-footnote-list {
  contain: layout style;
}

d-footnote-list > * {
  grid-column: text;
}

d-footnote-list a.footnote-backlink {
  color: rgba(0,0,0,0.3);
  padding-left: 0.5em;
}

</style>

<h3>Footnotes</h3>
<ol></ol>
</d-footnote-list> <d-citation-list style="display: none;"></d-citation-list> </d-appendix> <d-bibliography src="/assets/bibliography/"></d-bibliography> <d-article> <br> <br> </d-article> </div> <footer class="sticky-bottom mt-5" role="contentinfo"> <div class="container"> © Copyright 2026 Dao AI Lab. </div> </footer> <script src="about:blank" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4=" crossorigin="anonymous"></script> <script src="about:blank"></script> <script src="about:blank" integrity="sha256-NdbiivsvWt7VYCt6hYNT3h/th9vSTL4EDWeGs5SN3DA=" crossorigin="anonymous"></script> <script src="about:blank"></script> <script defer="" src="about:blank" integrity="sha256-ZgMyDAIYDYGxbcpJcfUnYwNevG/xi9OHKaR/8GK+jWc=" crossorigin="anonymous"></script> <script defer="" src="about:blank"></script> <script src="about:blank"></script> <script defer="" src="about:blank"></script> <script defer="" src="about:blank" type="text/javascript"></script> <script defer="" src="about:blank"></script> <script defer="" type="text/javascript" id="MathJax-script" src="about:blank" integrity="sha256-MASABpB4tYktI2Oitl4t+78w/lyA+D7b/s9GEP0JOGI=" crossorigin="anonymous"></script> <script src="about:blank"></script> <script defer="" src="about:blank" crossorigin="anonymous"></script> <script defer="" src="about:blank" type="text/javascript"></script> <script src="about:blank"></script> <script>
    addBackToTop();
  </script><div id="back-to-top" class="" style="position: fixed; bottom: 15px;"><svg viewBox="0 0 24 24"><path d="M7.41 15.41L12 10.83l4.59 4.58L18 14l-6-6-6 6z"></path></svg></div>  <div class="hiddendiv common"></div>
