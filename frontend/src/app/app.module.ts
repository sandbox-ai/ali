import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FormsModule } from '@angular/forms';

import { NbThemeModule, NbLayoutModule, NbCardModule, NbCheckboxModule } from '@nebular/theme';
import { NbEvaIconsModule } from '@nebular/eva-icons';
import { NbSearchModule } from '@nebular/theme'
import { NbIconModule } from '@nebular/theme';
import { NbButtonModule } from '@nebular/theme';
import { NbDialogModule } from '@nebular/theme';

import { AppRoutingModule } from './app-routing.module';

import { TermDialogComponent } from './components/term-dialog.component';

import { HttpClientModule } from '@angular/common/http';

import {NgxTypedJsModule} from 'ngx-typed-js';

import { TextStreamComponent } from './components/text-stream/text-stream.component';

@NgModule({
  declarations: [
    TermDialogComponent,
    AppComponent,
    TextStreamComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    NbThemeModule.forRoot({ name: 'default' }),
    FormsModule,
    NbLayoutModule,
    NbEvaIconsModule,
    NbSearchModule,
    NbIconModule,
    NbButtonModule,
    NbCardModule,
    NbCheckboxModule,
    NbDialogModule.forRoot(),
    AppRoutingModule,
    NgxTypedJsModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
