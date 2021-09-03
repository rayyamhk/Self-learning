const fs = require('fs');
const path = require('path');
const axios = require('axios');
const cheerio = require('cheerio');
const minify = require('html-minifier-terser').minify;

class BackupScheduler {
  constructor() {
    this.baseURL = 'https://thestandnews.com/';
    this.track_prev_date = undefined;
    this.sources = [
      'https://thestandnews.com/politics',
      'https://thestandnews.com/society',
      'https://thestandnews.com/court',
      'https://thestandnews.com/international',
      'https://thestandnews.com/finance',
      'https://thestandnews.com/culture',
      'https://thestandnews.com/art',
      'https://thestandnews.com/personal',
      'https://thestandnews.com/nature',
      'https://thestandnews.com/cosmos',
      'https://thestandnews.com/philosophy',
    ];

    if (!fs.existsSync('backup')) {
      fs.mkdirSync('backup');
    }

    if (!fs.existsSync('articles.json')) {
      this.articles = {};
    } else {
      const json = fs.readFileSync('articles.json');
      this.articles = JSON.parse(json);
    }

    let css = fs.readFileSync('style.css');
    this.css = css.toString();
  }

  async start() {
    for (const source of this.sources) {
      let page = 0;
      while (true) {
        const url = `${source}/page/${page}`;
        console.log(url);
        const { hasNextPage, articles } = await this._fetchArticlesFromDirectoryPage(url);

        if (articles.length !== 0) {
          for (let article of articles) {
            const { date, category, slug, articleUrl } = article;
            if (this.articles[slug]) {
              console.log('skip');
              continue;
            }

            console.time(slug);
            const { success, tags, html } = await this._exportHTML(articleUrl, slug);
            if (success) {
              this.articles[slug] = `${date}|${category}|${tags}`;

              const dir = this._createDirectoryByDate(date);
              const targetPath = path.join(dir, `${slug}.html`);

              fs.writeFileSync(targetPath, minify(html, {
                minifyCSS: true,
                collapseWhitespace: true,
              }));
            }
            console.timeEnd(slug)
          }
        } else {
          console.log(`Retry: ${url}`);
          continue;
        }

        fs.writeFileSync('articles.json', JSON.stringify(this.articles, null, 2));

        if (!hasNextPage) {
          break;
        }
        page++;
      }
      this.track_prev_date = undefined;
    }
  }

  async _fetchArticlesFromDirectoryPage(url) {
    const html = await this._getHTMLFromURL(url);

    if (!html) {
      return { hasNextPage: false, articles: [] };
    }

    const $ = cheerio.load(html);
    this._removeClassNameHash($.root()[0]);
    const articleCards = $('.ArticleList_articleList__listContainer > .ArticleItem_articleItem');
    const nextPage = $('.PagingArrows_pagingArrows__link--nextPage');

    const articles = []
    for (let card of articleCards) {
      const category = $(card).find('.CategoryTag_categoryTag').text();
      let time = $(card).find('.NewsBanner_newsBanner__timestamp').text();
      if (!time) {
        time = $(card).find('.BlogBanner_blogBanner__timestamp').text();
      }
      const date = this._getDateFromTimestamp(time);

      let articleUrl = $(card).find('.ArticleItem_articleItem__link')[0].attribs.href;
      articleUrl = this.baseURL + articleUrl;

      const slug = this._getSlugFromURL(articleUrl);
      articles.push({
        date,
        category,
        slug,
        articleUrl
      });
    }

    return { hasNextPage: nextPage.length > 0, articles };
  }

  async _getHTMLFromURL(url) {
    try {
      const resp  = await axios.get(url);
      return resp.data;
    } catch (err) {
      return null;
    }
  }

  _getSlugFromURL(url) {
    const url_split = url.split('/');
    return decodeURIComponent(url_split[url_split.length - 1]);
  }

  _getDateFromTimestamp(timestamp) {
    // Example: 2021/06/24 - 17:03
    let date_split = timestamp.split('-');
    let date;
    if (date_split.length === 1) {
      if (this.track_prev_date === undefined) {
        const today = new Date();
        const y = today.getFullYear();
        const m = (today.getMonth() + 1).toString().padStart(2, '0');
        const d = today.getDate().toString().padStart(2, '0');
        date = `${y}/${m}/${d}`;
        this.track_prev_date = date;
        return date;
      } else {
        return this.track_prev_date;
      }
    }
    date = date_split[0].trim();
    this.track_prev_date = date;
    return date;
  }

  async _exportHTML(url) {
    const html = await this._getHTMLFromURL(url);

    if (!html) {
      return { success: false, tags: '', html: '' };
    }

    const $ = cheerio.load(html);
    const article = $('article');
    if (article.length === 0) {
      return { success: false, tags: '', html: '' };
    }
    this._removeClassNameHash(article[0]);

    // remove unnecessary components
    article.find('ul.ArticleShareWidget_shareWidget').remove();
    article.find('.FontSizeSelection_container').remove();
    article.find('source').remove();

    const title = $('.ArticleDetail_article__title').text();

    let date = $('.ArticleDetail_article__publishedDate').text(); // Example: 2021/06/24 - 17:03
    date = this._getDateFromTimestamp(date); // Example: 2021/06/24

    const tags = $('.ArticleTags_root .ArticleTags_tagName');
    let str_tags = '';
    if (tags.length > 0) {
      for (const tag of tags) {
        str_tags += $(tag).text() + '-';
      }
      str_tags = str_tags.slice(0, -1);
    }

    let body = article.html();

    const html_str = `<html><head><title>${title}</title><style>${this.css}</style></head><body><article>${body}</article></body></html>`;

    return { success: true, tags: str_tags, html: html_str };
  }

  _removeClassNameHash(dom) {
    if (dom.attribs && dom.attribs.class && dom.attribs.class !== '') {
      // given that always append "__xxxxx" at the end of each className
      const hashRegex = /__[a-zA-Z0-9_-]{5}$/;
      const classNames = dom.attribs.class.split(' ');
      dom.attribs.class = classNames.map((className) => hashRegex.test(className) ? className.slice(0, -7) : className).join(' ');
    }
    dom.children && dom.children.forEach((child) => this._removeClassNameHash(child));
  }

  _createDirectoryByDate(date) {
    const [year, month, day] = date.split('/');
    const dayPath = path.join('./backup', year, month, day);
  
    if (!fs.existsSync(dayPath)) {
      fs.mkdirSync(dayPath, { recursive: true });
    }

    return dayPath;
  }
}

const scheduler = new BackupScheduler();
scheduler.start();